"""A second small fixture repo exercising the schema-v3 features: framework
routes (FastAPI / Flask / aiohttp / Express), MCP tools, call ambiguity,
receiver-aware resolution, an import cycle, dead code and git history.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

FW_FILES: dict[str, str] = {
    "app/__init__.py": "",
    "app/helpers.py": textwrap.dedent('''
        """Helpers in the app package."""


        def helper():
            return 1


        def format_name(x):
            return str(x).strip()
        ''').strip(),
    "lib/__init__.py": "",
    "lib/helpers.py": textwrap.dedent('''
        """Same-named helper in another package (ambiguity)."""


        def helper():
            return 2
        ''').strip(),
    "lib/tools.py": textwrap.dedent('''
        """Tools: unique names, fuzzy targets, dead code."""


        def unique_thing():
            return "u"


        def _check_value(v):
            return bool(v)


        def dumps(obj):
            return "not json"


        def never_used_zz():
            return None
        ''').strip(),
    "app/models.py": textwrap.dedent('''
        """A model class for trace tests."""


        class Repo:
            def save(self):
                return self._write()

            def _write(self):
                return True
        ''').strip(),
    "app/service.py": textwrap.dedent('''
        """Service layer."""

        from app.helpers import format_name
        from app.models import Repo
        import app.helpers


        def run_job():
            name = format_name("x")
            helper()
            repo = Repo()
            repo.save()
            return local_step(name)


        def local_step(name):
            return name.upper()
        ''').strip(),
    "other/__init__.py": "",
    "other/caller.py": textwrap.dedent('''
        """No imports: exercises unique / fuzzy / ambiguous / external."""

        import json


        def caller_fn():
            helper()
            unique_thing()
            check_value(3)
            return json.dumps({})
        ''').strip(),
    "web/__init__.py": "",
    "web/api.py": textwrap.dedent('''
        """FastAPI surface.

        Example that must NOT be detected: app.router.add_get("/fake", nothing)
        """
        from fastapi import FastAPI

        from app.service import run_job

        app = FastAPI()


        @app.get("/items/{item_id}")
        def read_item(item_id: int):
            return run_job()


        @app.post("/items")
        async def create_item():
            return {"ok": True}
        ''').strip(),
    "web/aio.py": textwrap.dedent('''
        from aiohttp import web


        async def handle_ping(request):
            return web.Response(text="pong")


        async def handle_echo(request):
            return web.Response(text="echo")


        def setup(app):
            app.router.add_get("/ping", handle_ping)
            app.router.add_route("POST", "/echo", handle_echo)
        ''').strip(),
    "web/fl.py": textwrap.dedent('''
        from flask import Flask

        app = Flask(__name__)


        @app.route("/hello", methods=["GET", "POST"])
        def hello():
            return "hi"
        ''').strip(),
    "web/server.js": textwrap.dedent('''
        const express = require('express');
        const app = express();

        function listUsers(req, res) {
          res.send([]);
        }

        app.get('/users', listUsers);
        app.post('/users', (req, res) => {
          res.send(1);
        });
        ''').strip(),
    "tools/__init__.py": "",
    "tools/mcp_srv.py": textwrap.dedent('''
        from fastmcp import FastMCP

        mcp = FastMCP("x")


        @mcp.tool()
        def add_numbers(a: int, b: int) -> int:
            return a + b


        @mcp.resource("res://greeting")
        def greeting() -> str:
            return "hi"


        @mcp.prompt()
        def review() -> str:
            return "review"
        ''').strip(),
    "cyc/__init__.py": "",
    "cyc/a.py": "import cyc.b\n\n\ndef fa():\n    return cyc.b.fb()\n",
    "cyc/b.py": "import cyc.a\n\n\ndef fb():\n    return 1\n",
    "tests/__init__.py": "",
    "tests/test_service.py": textwrap.dedent('''
        from app.service import run_job


        def test_run_job():
            assert run_job()
        ''').strip(),
}


def line_of(rel: str, needle: str) -> int:
    """1-based line of the first line containing `needle` in a fixture file."""
    for i, ln in enumerate((FW_FILES[rel] + "\n").splitlines(), start=1):
        if needle in ln:
            return i
    raise KeyError(needle)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def materialize(root: Path) -> None:
    for rel, content in FW_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content + "\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@test.test")
    _git(root, "config", "user.name", "test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "initial")
    # A second commit touching run_job so first_seen != last_changed
    svc = root / "app" / "service.py"
    svc.write_text(svc.read_text(encoding="utf-8").replace(
        'name = format_name("x")', 'name = format_name("xy")'), encoding="utf-8")
    _git(root, "commit", "-q", "-am", "tweak run_job")


def write_scip(root: Path) -> Path:
    """A synthetic SCIP index: run_job references local_step."""
    from docgraph.scip import Document, Occurrence, encode_index
    svc = "app/service.py"
    def_line = line_of(svc, "def local_step") - 1
    ref_line = line_of(svc, "return local_step(name)") - 1
    run_def = line_of(svc, "def run_job") - 1
    doc = Document(path=svc, language="python", occurrences=[
        Occurrence(line=def_line, symbol="scip-python python fw 0.1 app.service/local_step().", roles=1),
        Occurrence(line=run_def, symbol="scip-python python fw 0.1 app.service/run_job().", roles=1),
        Occurrence(line=ref_line, symbol="scip-python python fw 0.1 app.service/local_step().", roles=0),
        Occurrence(line=ref_line, symbol="local 3", roles=0),
    ])
    out = root / "index.scip"
    out.write_bytes(encode_index([doc]))
    return out
