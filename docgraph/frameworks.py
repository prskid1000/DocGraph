"""Framework route + MCP tool detection.

Table-driven regexes over a file's source text, run by `parse.py` after the
tree-sitter pass (so no per-language processor classes -- one pattern table
keyed by language). Each hit becomes a RawEdge of kind ROUTE or TOOL whose
`extra` carries the method / path / framework / kind and either the exact
handler qname (decorator style: the next definition below the decorator)
or a handler *name* the indexer resolves through the normal cascade
(registration style: `app.router.add_get("/x", handler)`).

Covered:
  Python  FastAPI / Flask / aiohttp RouteTableDef decorators
          (`@app.get("/x")`, `@bp.route("/x", methods=[...])`,
          `@routes.post("/x")`), aiohttp `router.add_get/add_post/...`,
          `web.get("/x", h)`, Starlette `Route("/x", h)`,
          FastAPI `add_api_route`, MCP `@mcp.tool/resource/prompt`.
  JS/TS   Express `app.get/post/...("/x", ..., handler)` (path must start
          with "/"), MCP `server.tool("name", ...)` / `registerTool`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HTTP = "get|post|put|delete|patch|head|options"

_PY_DECORATOR_ROUTE = re.compile(
    r"^\s*@\s*([A-Za-z_][\w.]*)\.(" + _HTTP + r"|route|api_route|websocket)\s*\(\s*"
    r"(?:path\s*=\s*|rule\s*=\s*)?[rbuRBU]?([\"'])(.*?)\3(.*)$"
)
_PY_ADD_ROUTE = re.compile(
    r"([A-Za-z_][\w.]*)\.add_(" + _HTTP + r"|route|view)\s*\(\s*"
    r"(?:([\"'])([A-Za-z*]+)\3\s*,\s*)?([\"'])(.*?)\5\s*,\s*([A-Za-z_][\w.]*)"
)
_PY_WEB_ROUTE = re.compile(
    r"\bweb\.(" + _HTTP + r"|route|view)\s*\(\s*"
    r"(?:([\"'])([A-Za-z*]+)\2\s*,\s*)?([\"'])(.*?)\4\s*,\s*([A-Za-z_][\w.]*)"
)
_PY_STARLETTE_ROUTE = re.compile(
    r"\bRoute\s*\(\s*([\"'])(/.*?)\1\s*,\s*(?:endpoint\s*=\s*)?([A-Za-z_][\w.]*)(.*)$"
)
_PY_ADD_API_ROUTE = re.compile(
    r"\.add_api_route\s*\(\s*([\"'])(.*?)\1\s*,\s*([A-Za-z_][\w.]*)(.*)$"
)
_PY_MCP = re.compile(
    r"^\s*@\s*([A-Za-z_][\w.]*)\.(tool|resource|prompt)\s*(?:\((.*))?$"
)
_JS_EXPRESS = re.compile(
    r"\b([A-Za-z_$][\w$]*)\.(get|post|put|delete|patch|all|options|head)\s*\(\s*"
    r"([\"'`])(/[^\"'`]*)\3\s*(.*)$"
)
_JS_MCP = re.compile(
    r"\b([A-Za-z_$][\w$]*)\.(tool|registerTool|resource|registerResource|prompt|registerPrompt)"
    r"\s*\(\s*([\"'`])(.*?)\3"
)
_METHODS_KW = re.compile(r"methods\s*=\s*[\[(]([^\])]*)[\])]")
_NAME_KW = re.compile(r"\bname\s*=\s*[\"']([^\"']+)[\"']")
_FIRST_STR = re.compile(r"^\s*[rbuRBU]?[\"']([^\"']+)[\"']")
_IDENT_TAIL = re.compile(r"([A-Za-z_$][\w$.]*)\s*\)?\s*;?\s*$")

_JS_LANGS = {"javascript", "typescript", "tsx"}


@dataclass
class Hit:
    kind: str                 # "ROUTE" | "TOOL"
    line: int
    name: str                 # "GET /api/x" or tool name
    framework: str
    method: str = ""
    path: str = ""
    tool_kind: str = ""       # tool | resource | prompt
    handler_qname: str | None = None
    handler_name: str = ""
    extra: dict = field(default_factory=dict)


def _py_framework(source: str) -> str:
    head = source[:20000]
    for fw, pat in (("fastapi", r"\bfastapi\b"), ("flask", r"\bflask\b"),
                    ("aiohttp", r"\baiohttp\b"), ("starlette", r"\bstarlette\b"),
                    ("django", r"\bdjango\b")):
        if re.search(r"(?:^|\n)\s*(?:from|import)\s+[^\n]*" + pat, head, re.IGNORECASE):
            return fw
    return "python"


def _next_def(defs: list[tuple[str, str, int, int]], line: int, window: int = 15) -> str | None:
    best = None
    for kind, qname, s, _e in defs:
        if kind not in ("function", "method"):
            continue
        if line < s <= line + window and (best is None or s < best[0]):
            best = (s, qname)
    return best[1] if best else None


def _enclosing(defs: list[tuple[str, str, int, int]], line: int) -> str | None:
    best = None
    for kind, qname, s, e in defs:
        if kind not in ("function", "method"):
            continue
        if s <= line <= e and (best is None or (e - s) < best[0]):
            best = (e - s, qname)
    return best[1] if best else None


def _js_handler(rest: str) -> str:
    """Last argument of an Express registration if it is a plain
    identifier (`ctrl.list`), else "" (inline arrow / function)."""
    depth = 0
    buf = ""
    for ch in rest:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        buf += ch
    parts = [p.strip() for p in buf.split(",") if p.strip()]
    if not parts:
        return ""
    last = parts[-1]
    if re.fullmatch(r"[A-Za-z_$][\w$.]*", last) and last not in ("async", "function"):
        return last
    return ""


def detect(source: str, language: str,
           defs: list[tuple[str, str, int, int]]) -> list[Hit]:
    """defs: [(kind, qname, line_start, line_end)] of the file's entities."""
    hits: list[Hit] = []
    if language == "python":
        fw = _py_framework(source)
        for i, text in enumerate(source.splitlines(), start=1):
            if "@" in text:
                m = _PY_DECORATOR_ROUTE.match(text)
                if m:
                    owner, verb, _q, path, rest = m.groups()
                    if verb in ("route", "api_route"):
                        mm = _METHODS_KW.search(rest)
                        methods = re.findall(r"[A-Za-z]+", mm.group(1)) if mm else []
                        method = ",".join(x.upper() for x in methods) or ("GET" if fw == "flask" else "ANY")
                    elif verb == "websocket":
                        method = "WS"
                    else:
                        method = verb.upper()
                    framework = fw
                    if framework == "python":
                        framework = "flask" if verb == "route" else "fastapi"
                    if owner.split(".")[-1] == "routes" and framework == "python":
                        framework = "aiohttp"
                    hits.append(Hit("ROUTE", i, f"{method} {path}", framework, method, path,
                                    handler_qname=_next_def(defs, i)))
                    continue
                m = _PY_MCP.match(text)
                if m:
                    owner, kind, rest = m.groups()
                    if owner.split(".")[0] in ("pytest", "functools", "typing"):
                        continue
                    qn = _next_def(defs, i)
                    rest = rest or ""
                    nm = _NAME_KW.search(rest)
                    first = _FIRST_STR.match(rest)
                    uri = first.group(1) if (first and kind in ("resource", "prompt")) else ""
                    name = nm.group(1) if nm else (uri or (qn.rsplit("::", 1)[-1] if qn else ""))
                    if not name:
                        continue
                    hits.append(Hit("TOOL", i, name, "mcp", tool_kind=kind,
                                    handler_qname=qn, extra={"uri": uri} if uri else {}))
                    continue
            for rx, fwname in ((_PY_ADD_ROUTE, "aiohttp"), (_PY_WEB_ROUTE, "aiohttp")):
                for m in rx.finditer(text):
                    g = m.groups()
                    if rx is _PY_ADD_ROUTE:
                        _owner, verb, _q1, explicit, _q2, path, handler = g
                    else:
                        verb, _q1, explicit, _q2, path, handler = g
                    method = (explicit or ("ANY" if verb in ("route", "view") else verb)).upper()
                    hits.append(Hit("ROUTE", i, f"{method} {path}", fwname, method, path,
                                    handler_name=handler.rsplit(".", 1)[-1]))
            m = _PY_STARLETTE_ROUTE.search(text)
            if m and "add_api_route" not in text:
                _q, path, handler, rest = m.groups()
                mm = _METHODS_KW.search(rest or "")
                methods = re.findall(r"[A-Za-z]+", mm.group(1)) if mm else []
                method = ",".join(x.upper() for x in methods) or "ANY"
                hits.append(Hit("ROUTE", i, f"{method} {path}", "starlette", method, path,
                                handler_name=handler.rsplit(".", 1)[-1]))
            m = _PY_ADD_API_ROUTE.search(text)
            if m:
                _q, path, handler, rest = m.groups()
                mm = _METHODS_KW.search(rest or "")
                methods = re.findall(r"[A-Za-z]+", mm.group(1)) if mm else []
                method = ",".join(x.upper() for x in methods) or "GET"
                hits.append(Hit("ROUTE", i, f"{method} {path}", "fastapi", method, path,
                                handler_name=handler.rsplit(".", 1)[-1]))
    elif language in _JS_LANGS:
        for i, text in enumerate(source.splitlines(), start=1):
            m = _JS_EXPRESS.search(text)
            if m:
                _owner, verb, _q, path, rest = m.groups()
                method = "ANY" if verb == "all" else verb.upper()
                handler = _js_handler(rest.lstrip(", "))
                enc = _enclosing(defs, i)
                hits.append(Hit("ROUTE", i, f"{method} {path}", "express", method, path,
                                handler_name=handler.rsplit(".", 1)[-1] if handler else "",
                                handler_qname=None if handler else enc,
                                extra={} if handler else {"inline": True}))
                continue
            m = _JS_MCP.search(text)
            if m:
                _owner, kind, _q, name = m.groups()
                kind = kind.replace("register", "").lower()
                if not name:
                    continue
                hits.append(Hit("TOOL", i, name, "mcp", tool_kind=kind,
                                handler_qname=_enclosing(defs, i), extra={"inline": True}))
    # De-dup (a line can match two registration forms)
    seen: set[tuple[str, int, str]] = set()
    out: list[Hit] = []
    for h in hits:
        key = (h.kind, h.line, h.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
