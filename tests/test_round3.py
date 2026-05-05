"""Tests for round 3: cross-encoder reranker and scope-aware resolution."""
from __future__ import annotations

import gc
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer
from docgraph.rerank import Reranker


# --- Reranker (mocked to avoid model download) ------------------------


def test_reranker_score_orders_results():
    """Verify rerank.rerank() reorders correctly given known scores."""
    rer = Reranker()
    items = [
        {"snippet": "a thing about cats", "score": 0.5},
        {"snippet": "exactly what you asked", "score": 0.5},
        {"snippet": "unrelated stuff", "score": 0.5},
    ]
    fake_scores = [0.1, 0.95, 0.2]
    with patch.object(rer, "score", return_value=fake_scores):
        out = rer.rerank("query", items, top_k=3)
    assert out[0]["snippet"] == "exactly what you asked"
    assert out[0]["rerank_score"] == pytest.approx(0.95)


def test_reranker_keeps_tail_unsorted():
    """Items past top_k aren't rescored; they tail the reranked head."""
    rer = Reranker()
    items = [
        {"snippet": f"item{i}", "score": 1.0 - i * 0.01}
        for i in range(8)
    ]
    fake_scores = [0.1, 0.9]  # only first 2 get reranked
    with patch.object(rer, "score", return_value=fake_scores):
        out = rer.rerank("q", items, top_k=2)
    # Reranked head: item1 (0.9) before item0 (0.1)
    assert out[0]["snippet"] == "item1"
    # Unranked tail starts at index 2 with item2
    assert out[2]["snippet"] == "item2"


def test_reranker_empty_items():
    rer = Reranker()
    assert rer.rerank("q", [], top_k=10) == []


def test_search_with_rerank_falls_back_silently(retriever):
    """If the reranker can't load (no model, offline), search should still
    return results from the bi-encoder ranker."""
    with patch.object(Reranker, "score", side_effect=RuntimeError("model unavailable")):
        results = retriever.search("login", limit=5, rerank=True)
    assert results  # no exception, still got hits


# --- Scope-aware resolution -------------------------------------------


@pytest.fixture(scope="module")
def scope_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two files define a function with the same name; only one is imported
    by the caller. Scope-aware resolution should pick the imported one."""
    root = tmp_path_factory.mktemp("docgraph_scope")
    (root / "real.py").write_text(textwrap.dedent('''
        """The 'real' authenticator."""

        def authenticate(user: str) -> str:
            return f"real-token-for-{user}"
    ''').strip() + "\n")
    (root / "fake.py").write_text(textwrap.dedent('''
        """A homonym in an unrelated module — should NOT be the call target."""

        def authenticate(user: str) -> str:
            return "fake"
    ''').strip() + "\n")
    (root / "caller.py").write_text(textwrap.dedent('''
        """Imports real, calls authenticate."""

        from real import authenticate


        def login_flow(u: str) -> str:
            return authenticate(u)
    ''').strip() + "\n")
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=root, check=False)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=False)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=False)
    subprocess.run(["git", "add", "."], cwd=root, check=False)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=False)
    return root


def test_scope_aware_calls_resolves_to_imported_file(scope_repo: Path):
    cfg = load_config(scope_repo)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    Indexer(cfg, db, embedder=embedder).index_all(incremental=False)
    del db
    gc.collect()
    db = GraphDB(cfg.db_path, read_only=True)

    # The CALLS edge from login_flow should target authenticate in real.py,
    # not the homonym in fake.py.
    rows = db.fetch_all(
        "MATCH (a:Function)-[:CALLS]->(b:Function) "
        "WHERE a.name = 'login_flow' AND b.name = 'authenticate' "
        "RETURN b.file AS file"
    )
    target_files = [r["file"] for r in rows]
    assert target_files, "expected a CALLS edge from login_flow → authenticate"
    assert "real.py" in target_files[0], (
        f"scope-aware resolution should prefer the imported file, got {target_files}"
    )
