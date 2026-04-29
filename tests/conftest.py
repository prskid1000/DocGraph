"""Session-scoped fixture: build a tiny mock repo and index it once.

The model download + embedding pass is the slow part (~20s cold). Tests share
one indexed DB read-only via a Retriever fixture.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer
from docgraph.retrieve import Retriever


# --- Mock repo content ---------------------------------------------------

REPO_FILES: dict[str, str] = {
    "src/__init__.py": "",
    "src/auth.py": textwrap.dedent('''
        """Authentication utilities."""

        class TokenError(Exception):
            """Raised when a token is invalid."""
            pass


        class Authenticator:
            """Validates user credentials and issues tokens."""

            def __init__(self, secret: str):
                self.secret = secret

            def login(self, username: str, password: str) -> str:
                """Verify password and return a session token."""
                if not self._check_password(username, password):
                    raise TokenError("bad credentials")
                return self._issue_token(username)

            def _check_password(self, username: str, password: str) -> bool:
                return bool(username) and bool(password)

            def _issue_token(self, username: str) -> str:
                return f"token-for-{username}"
        ''').strip(),
    "src/api.py": textwrap.dedent('''
        """HTTP API surface that uses the authenticator."""

        from src.auth import Authenticator, TokenError


        def make_handler(secret: str):
            """Build the request handler closure."""
            auth = Authenticator(secret)
            def handle(req: dict) -> dict:
                try:
                    token = auth.login(req["user"], req["pw"])
                    return {"ok": True, "token": token}
                except TokenError:
                    return {"ok": False}
            return handle
        ''').strip(),
    "src/utils.py": textwrap.dedent('''
        """Misc utilities used across the codebase."""


        def slugify(text: str) -> str:
            """Lowercase + dasherize."""
            return text.strip().lower().replace(" ", "-")


        def chunked(seq: list, n: int) -> list:
            return [seq[i:i+n] for i in range(0, len(seq), n)]
        ''').strip(),
    "tests/test_auth.py": textwrap.dedent('''
        """Tests for the authenticator."""

        from src.auth import Authenticator, TokenError


        def test_login():
            a = Authenticator("s")
            assert a.login("u", "p")


        def test_login_failure():
            a = Authenticator("s")
            try:
                a.login("", "")
            except TokenError:
                return
            assert False, "should raise"


        def test_slugify():
            from src.utils import slugify
            assert slugify("Hello World") == "hello-world"
        ''').strip(),
    # A deliberately long function (>1500 chars body) so chunking kicks in.
    "src/big.py": textwrap.dedent('''
        """A module with a long function for chunking tests."""


        def process_pipeline(records):
            """Run a 5-stage data pipeline on the input records.

            Stages: validate, normalize, enrich, transform, persist.
            Each stage is documented inline below.
            """
            # ============= STAGE 1: validate =============
            validated = []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                if "id" not in rec:
                    continue
                if not rec.get("name"):
                    continue
                validated.append(rec)
            # ============= STAGE 2: normalize =============
            normalized = []
            for rec in validated:
                rec = dict(rec)
                rec["name"] = rec["name"].strip().lower()
                rec.setdefault("tags", [])
                rec["tags"] = [t.strip().lower() for t in rec["tags"]]
                normalized.append(rec)
            # ============= STAGE 3: enrich =============
            enriched = []
            for rec in normalized:
                rec = dict(rec)
                rec["enriched_at"] = "2024-01-01"
                rec["region"] = rec.get("region", "global")
                rec["score"] = sum(len(t) for t in rec["tags"])
                enriched.append(rec)
            # ============= STAGE 4: transform =============
            transformed = []
            for rec in enriched:
                rec = dict(rec)
                rec["display_name"] = rec["name"].title()
                rec["category"] = "low" if rec["score"] < 5 else "high"
                transformed.append(rec)
            # ============= STAGE 5: persist =============
            persisted = []
            for rec in transformed:
                # Pretend to write to a database here
                rec["persisted"] = True
                persisted.append(rec)
            return persisted
    ''').strip(),
}


def _materialize_repo(root: Path) -> None:
    for rel, content in REPO_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content + "\n", encoding="utf-8")
    # git init so co-changed history exists (even if empty)
    subprocess.run(["git", "init", "-q"], cwd=root, check=False)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=root, check=False)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=False)
    subprocess.run(["git", "add", "."], cwd=root, check=False)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=False)


@pytest.fixture(scope="session")
def repo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("docgraph_repo")
    _materialize_repo(root)
    return root


@pytest.fixture(scope="session")
def indexed(repo_dir: Path):
    """Run a full index, then drop the writer connection and yield a fresh
    read-only connection. (Kuzu writer connections don't see their own
    writes via subsequent reads in the same session.)"""
    import gc

    cfg = load_config(repo_dir)
    writer = GraphDB(cfg.db_path, embedding_dim=384)
    writer.init_schema()
    embedder = Embedder(cfg.embedding_model)
    indexer = Indexer(cfg, writer, embedder=embedder)
    stats = indexer.index_all(incremental=False)
    assert stats["errors"] == 0, f"index errors: {stats}"
    # Release writer so a read-only connection can take over.
    del writer, indexer
    gc.collect()
    reader = GraphDB(cfg.db_path, read_only=True)
    yield cfg, reader, embedder, stats


@pytest.fixture(scope="session")
def retriever(indexed) -> Retriever:
    _cfg, db, embedder, _stats = indexed
    return Retriever(db, embedder)
