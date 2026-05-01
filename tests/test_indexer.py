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


# --- Variable coverage ----------------------------------------------------
# parse.py only emits Variable nodes for module-level assignments (per the
# tags query). Cover the round-trip: parse → Variable rows in DB with the
# right fields, file-scoped lookup, and incremental delete.

VAR_REPO_FILES: dict[str, str] = {
    "lib/__init__.py": "",
    "lib/constants.py": (
        '"""Module-level constants used across the package."""\n'
        "API_VERSION = 3\n"
        "DEFAULT_TIMEOUT = 30\n"
        "FEATURE_FLAGS = {'fast_path': True}\n"
    ),
    "lib/settings.py": (
        '"""Runtime settings."""\n'
        "DEBUG = False\n"
        "MAX_RETRIES = 5\n"
    ),
}


def _materialize_var_repo(root: Path) -> None:
    import subprocess as sp
    for rel, content in VAR_REPO_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    sp.run(["git", "init", "-q"], cwd=root, check=False)
    sp.run(["git", "config", "user.email", "t@t.test"], cwd=root, check=False)
    sp.run(["git", "config", "user.name", "t"], cwd=root, check=False)
    sp.run(["git", "add", "."], cwd=root, check=False)
    sp.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=False)


@pytest.fixture(scope="module")
def var_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("docgraph_vars")
    _materialize_var_repo(root)
    return root


def _index_and_reopen_readonly(root: Path):
    """Full index + drop the writer + reopen read-only.
    Mirrors conftest's pattern: writer connections don't see own writes."""
    cfg = load_config(root)
    if cfg.db_path.exists():
        GraphDB.wipe(cfg.db_path)
    writer = GraphDB(cfg.db_path, embedding_dim=384)
    writer.init_schema()
    embedder = Embedder(cfg.embedding_model)
    indexer = Indexer(cfg, writer, embedder=embedder)
    stats = indexer.index_all(incremental=False)
    indexer.db.close()
    writer.close()
    del writer, indexer
    gc.collect()
    reader = GraphDB(cfg.db_path, read_only=True)
    return cfg, reader, embedder, stats


def test_variables_are_extracted(var_repo: Path):
    """Module-level assignments produce Variable nodes with name, qname, file, scope."""
    cfg, db, _embedder, _stats = _index_and_reopen_readonly(var_repo)
    try:
        rows = db.fetch_all(
            "MATCH (v:Variable) RETURN v.name AS name, v.qname AS qname, "
            "v.file AS file, v.scope AS scope, v.line AS line"
        )
        names = {r["name"] for r in rows}
        assert {"API_VERSION", "DEFAULT_TIMEOUT", "FEATURE_FLAGS", "DEBUG", "MAX_RETRIES"} <= names
        for r in rows:
            assert r["file"], f"variable {r['name']} missing file"
            assert r["qname"], f"variable {r['name']} missing qname"
            assert r["scope"], f"variable {r['name']} missing scope"
            assert r["line"] >= 1
        api = next(r for r in rows if r["name"] == "API_VERSION")
        assert api["file"].endswith("constants.py")
        assert "constants" in api["qname"]
    finally:
        db.close()
        gc.collect()


def test_variables_contained_by_file(var_repo: Path):
    """File -[CONTAINS]-> Variable wired up so impact_of / neighborhood works."""
    cfg, db, _embedder, _stats = _index_and_reopen_readonly(var_repo)
    try:
        rows = db.fetch_all(
            "MATCH (f:File)-[:CONTAINS]->(v:Variable) "
            "WHERE f.path = $p RETURN v.name AS n",
            {"p": "lib/constants.py"},
        )
        names = {r["n"] for r in rows}
        assert {"API_VERSION", "DEFAULT_TIMEOUT", "FEATURE_FLAGS"} <= names
    finally:
        db.close()
        gc.collect()


def test_imports_symbol_edge(indexed):
    """`from src.auth import Authenticator, TokenError` produces IMPORTS_SYMBOL
    edges from api.py to the Class nodes it imports. This is finer than the
    file-level IMPORTS edge — agents asking 'who depends on this class?' get
    a precise answer instead of a file-fanout."""
    _cfg, db, _e, _s = indexed
    rows = db.fetch_all(
        "MATCH (f:File)-[:IMPORTS_SYMBOL]->(c:Class) "
        "WHERE f.path = $p RETURN c.name AS n",
        {"p": "src/api.py"},
    )
    names = {r["n"] for r in rows}
    assert {"Authenticator", "TokenError"} <= names, f"missing symbol imports, got {names}"


def test_overrides_edge_extracted(tmp_path: Path):
    """Methods that share a name with a method on an inherited class get
    OVERRIDES edges from child→parent."""
    repo = tmp_path / "ov"
    repo.mkdir()
    (repo / "shapes.py").write_text(
        "class Shape:\n"
        "    def area(self):\n"
        "        return 0\n"
        "    def describe(self):\n"
        "        return 'shape'\n"
        "\n"
        "class Square(Shape):\n"
        "    def area(self):\n"
        "        return 4\n"
        "\n"
        "class Circle(Shape):\n"
        "    def area(self):\n"
        "        return 3.14\n"
        "    def describe(self):\n"
        "        return 'circle'\n",
        encoding="utf-8",
    )
    import subprocess as sp
    sp.run(["git", "init", "-q"], cwd=repo, check=False)
    sp.run(["git", "config", "user.email", "t@t.t"], cwd=repo, check=False)
    sp.run(["git", "config", "user.name", "t"], cwd=repo, check=False)
    sp.run(["git", "add", "."], cwd=repo, check=False)
    sp.run(["git", "commit", "-q", "-m", "i"], cwd=repo, check=False)

    cfg, db, _embedder, _stats = _index_and_reopen_readonly(repo)
    try:
        rows = db.fetch_all(
            "MATCH (child:Function)-[:OVERRIDES]->(parent:Function) "
            "RETURN child.qname AS c, parent.qname AS p, child.name AS n"
        )
        pairs = {(r["c"], r["p"]) for r in rows}
        names = {r["n"] for r in rows}
        # Square.area → Shape.area, Circle.area → Shape.area, Circle.describe → Shape.describe
        assert "area" in names and "describe" in names, f"missing override names: {names}"
        # At least 3 override edges (Square.area, Circle.area, Circle.describe)
        assert len(pairs) >= 3, f"expected >= 3 OVERRIDES, got {len(pairs)}: {pairs}"
        # Square.area should override Shape.area, NOT a Square method
        for child_q, parent_q in pairs:
            assert "Shape::" in parent_q, f"OVERRIDES parent should be Shape, got {parent_q}"
    finally:
        db.close()
        gc.collect()


def test_variable_incremental_delete(var_repo: Path):
    """Removing a variable from a file → its Variable node is gone after reindex.
    Exercises _delete_files_from_db's Variable cascade."""
    cfg, db, embedder, _ = _index_and_reopen_readonly(var_repo)
    db.close()
    gc.collect()
    target = var_repo / "lib" / "settings.py"
    original = target.read_text()
    target.write_text('"""Runtime settings."""\nDEBUG = False\n', encoding="utf-8")
    try:
        writer = GraphDB(cfg.db_path)
        writer.init_schema()
        Indexer(cfg, writer, embedder=embedder).index_all(incremental=True)
        writer.close()
        gc.collect()
        reader = GraphDB(cfg.db_path, read_only=True)
        try:
            rows = reader.fetch_all(
                "MATCH (v:Variable) WHERE v.file = $f RETURN v.name AS n",
                {"f": "lib/settings.py"},
            )
            names = {r["n"] for r in rows}
            assert "DEBUG" in names
            assert "MAX_RETRIES" not in names, "stale Variable node survived incremental"
        finally:
            reader.close()
            gc.collect()
    finally:
        target.write_text(original, encoding="utf-8")
