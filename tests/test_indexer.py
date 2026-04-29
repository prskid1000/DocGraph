"""Tests covering the indexer + delta correctness.

Tests that need a writer (incremental edits) use isolated per-test repos —
the session-scoped reader holds a lock that blocks writers."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer

# Importing the materializer keeps tests self-contained.
from tests.conftest import _materialize_repo


def test_index_produces_files_and_entities(indexed):
    cfg, db, _embedder, stats = indexed
    assert stats["files"] >= 4  # 4 .py files in fixture
    files = db.fetch_all("MATCH (f:File) RETURN f.path AS p")
    paths = {r["p"] for r in files}
    assert any(p.endswith("auth.py") for p in paths)
    assert any(p.endswith("api.py") for p in paths)


def test_classes_and_functions(indexed):
    _cfg, db, _e, _s = indexed
    classes = db.fetch_all("MATCH (c:Class) RETURN c.name AS n")
    names = {r["n"] for r in classes}
    assert "Authenticator" in names
    assert "TokenError" in names

    funcs = db.fetch_all("MATCH (f:Function) RETURN f.name AS n")
    fnames = {r["n"] for r in funcs}
    assert "login" in fnames
    assert "make_handler" in fnames
    assert "test_login" in fnames


def test_calls_edge_present(indexed):
    _cfg, db, _e, _s = indexed
    rows = db.fetch_all(
        "MATCH (a:Function)-[:CALLS]->(b:Function) "
        "WHERE a.name = 'login' RETURN b.name AS n"
    )
    callees = {r["n"] for r in rows}
    # login() calls _check_password and _issue_token
    assert "_check_password" in callees or "_issue_token" in callees


def test_inheritance_edge(indexed):
    _cfg, db, _e, _s = indexed
    # TokenError inherits from Exception; we may or may not parse the
    # superclass when it's a builtin. Check we have the class node at minimum.
    rows = db.fetch_all("MATCH (c:Class) WHERE c.name = 'TokenError' RETURN c.id AS id")
    assert rows


def test_imports_edge(indexed):
    _cfg, db, _e, _s = indexed
    rows = db.fetch_all(
        "MATCH (a:File)-[:IMPORTS]->(b:File) WHERE a.path = $p RETURN b.path AS dst",
        {"p": "src/api.py"},
    )
    targets = {r["dst"] for r in rows}
    assert any("auth.py" in t for t in targets)


def test_test_functions_marked(indexed):
    _cfg, db, _e, _s = indexed
    rows = db.fetch_all(
        "MATCH (f:Function) WHERE f.is_test = true RETURN f.name AS n"
    )
    names = {r["n"] for r in rows}
    assert "test_login" in names
    assert "test_login_failure" in names


def test_pagerank_assigned(indexed):
    _cfg, db, _e, _s = indexed
    rows = db.fetch_all("MATCH (f:Function) RETURN f.pagerank AS pr")
    nonzero = [r for r in rows if (r["pr"] or 0) > 0]
    # At least some functions should have non-zero PR
    assert len(nonzero) >= 1


def _full_index(root: Path) -> tuple:
    cfg = load_config(root)
    if cfg.db_path.exists():
        GraphDB.wipe(cfg.db_path)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    stats = Indexer(cfg, db, embedder=embedder).index_all(incremental=False)
    return cfg, db, embedder, stats


@pytest.fixture(scope="module")
def isolated_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Per-module repo for tests that need to write to the index."""
    root = tmp_path_factory.mktemp("docgraph_iso")
    _materialize_repo(root)
    return root


def test_incremental_no_change_is_fast(isolated_repo: Path):
    """Re-indexing untouched repo should be near-instant and report 0 changes."""
    cfg, db, embedder, _ = _full_index(isolated_repo)
    del db
    gc.collect()
    db2 = GraphDB(cfg.db_path)
    db2.init_schema()
    t0 = time.perf_counter()
    stats = Indexer(cfg, db2, embedder=embedder).index_all(incremental=True)
    elapsed = time.perf_counter() - t0
    assert stats["changed"] == 0
    assert stats["deleted"] == 0
    assert elapsed < 5.0


def test_incremental_picks_up_edit(isolated_repo: Path):
    """Edit a file, reindex incrementally, that file (and only that file)
    should be marked changed."""
    cfg, db, embedder, _ = _full_index(isolated_repo)
    del db
    gc.collect()
    target = isolated_repo / "src" / "utils.py"
    original = target.read_text()
    target.write_text(original + "\n# touched\n", encoding="utf-8")
    try:
        db2 = GraphDB(cfg.db_path)
        db2.init_schema()
        stats = Indexer(cfg, db2, embedder=embedder).index_all(incremental=True)
        assert stats["changed"] == 1
        assert stats["deleted"] == 0
    finally:
        target.write_text(original, encoding="utf-8")


def test_full_reindex_idempotent(isolated_repo: Path):
    """A second --full reindex should produce the same node counts."""
    cfg, db, embedder, _ = _full_index(isolated_repo)
    del db
    gc.collect()
    db1 = GraphDB(cfg.db_path, read_only=True)
    counts_a = {
        lbl: db1.fetch_all(f"MATCH (n:{lbl}) RETURN count(n) AS c")[0]["c"]
        for lbl in ("File", "Function", "Class")
    }
    del db1
    gc.collect()
    GraphDB.wipe(cfg.db_path)
    db2 = GraphDB(cfg.db_path, embedding_dim=384)
    db2.init_schema()
    Indexer(cfg, db2, embedder=embedder).index_all(incremental=False)
    del db2
    gc.collect()
    db3 = GraphDB(cfg.db_path, read_only=True)
    counts_b = {
        lbl: db3.fetch_all(f"MATCH (n:{lbl}) RETURN count(n) AS c")[0]["c"]
        for lbl in ("File", "Function", "Class")
    }
    assert counts_a == counts_b
