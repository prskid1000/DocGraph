"""Multi-repo / cross-repo indexing test."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer, walk_files


@pytest.fixture(scope="module")
def two_repos(tmp_path_factory: pytest.TempPathFactory):
    a = tmp_path_factory.mktemp("repo_a")
    b = tmp_path_factory.mktemp("repo_b")
    (a / "alpha.py").write_text(textwrap.dedent('''
        """Alpha module."""

        def alpha_func():
            return 1
    ''').strip() + "\n")
    (b / "beta.py").write_text(textwrap.dedent('''
        """Beta module."""

        def beta_func():
            return 2
    ''').strip() + "\n")
    return a, b


def test_walk_files_multi_root(two_repos):
    a, b = two_repos
    cfg = load_config(a, extra_roots=[b])
    files = walk_files(cfg)
    rels = {rel for _path, rel in files}
    # Both repos contribute, paths prefixed with basename
    assert any(rel.endswith("/alpha.py") for rel in rels)
    assert any(rel.endswith("/beta.py") for rel in rels)


def test_walk_files_single_root_no_prefix(two_repos):
    a, _b = two_repos
    cfg = load_config(a, extra_roots=[])  # explicitly clear any persisted list
    files = walk_files(cfg)
    rels = {rel for _path, rel in files}
    # Single-root: no prefix
    assert "alpha.py" in rels


def test_index_multi_repo(two_repos):
    import gc
    a, b = two_repos
    cfg = load_config(a, extra_roots=[b])
    if cfg.db_path.exists():
        GraphDB.wipe(cfg.db_path)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    stats = Indexer(cfg, db, embedder=embedder).index_all(incremental=False)
    assert stats["errors"] == 0
    # Reopen as reader (Kuzu writer doesn't see its own writes)
    del db
    gc.collect()
    db = GraphDB(cfg.db_path, read_only=True)

    file_paths = {r["p"] for r in db.fetch_all("MATCH (f:File) RETURN f.path AS p")}
    # Both repos' files should appear, with their basename prefix
    assert any("alpha.py" in p for p in file_paths)
    assert any("beta.py" in p for p in file_paths)
    assert any("/" in p for p in file_paths), "multi-repo paths should include a `<repo>/` prefix"

    # Functions from both repos
    fnames = {r["n"] for r in db.fetch_all("MATCH (f:Function) RETURN f.name AS n")}
    assert "alpha_func" in fnames
    assert "beta_func" in fnames


def test_path_for_resolves_prefixed(two_repos):
    a, b = two_repos
    cfg = load_config(a, extra_roots=[b])
    # path_for should map the prefixed logical path back to the right disk loc
    rel = f"{b.name}/beta.py"
    resolved = cfg.path_for(rel)
    assert resolved.exists()
    assert resolved.read_text().startswith('"""Beta module."""')
