"""Workspace tests: slug-based + path-based + file-prefix resolution,
default-root semantics, list/slugs/default_slug, writer take/release.
"""
from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder, clear_model_cache
from docgraph.index import Indexer
from docgraph.workspace import Workspace, slug_for_root


def _tiny_index(root: Path) -> None:
    """Create a tiny indexed repo at `root`."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "a.py").write_text("def hello(): return 1\n", encoding="utf-8")
    cfg = load_config(root)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    indexer = Indexer(cfg, db, embedder=embedder)
    indexer.index_all(incremental=False)
    indexer.db.close()
    db.close()


@pytest.fixture(scope="module")
def two_roots(tmp_path_factory):
    a = tmp_path_factory.mktemp("ws_a")
    b = tmp_path_factory.mktemp("ws_b")
    _tiny_index(a)
    _tiny_index(b)
    cfg_a = load_config(a)
    cfg_b = load_config(b)
    yield cfg_a, cfg_b
    shutil.rmtree(a, ignore_errors=True)
    shutil.rmtree(b, ignore_errors=True)


def test_workspace_default_is_first(two_roots):
    cfg_a, cfg_b = two_roots
    with Workspace([cfg_a, cfg_b]) as ws:
        assert ws.default().cfg.repo_root == cfg_a.repo_root
        assert ws.default_slug() == slug_for_root(cfg_a.repo_root)


def test_resolve_by_path(two_roots):
    cfg_a, cfg_b = two_roots
    with Workspace([cfg_a, cfg_b]) as ws:
        assert ws.resolve(cfg_b.repo_root).cfg.repo_root == cfg_b.repo_root
        assert ws.resolve(str(cfg_a.repo_root)).cfg.repo_root == cfg_a.repo_root


def test_resolve_by_slug(two_roots):
    cfg_a, cfg_b = two_roots
    with Workspace([cfg_a, cfg_b]) as ws:
        slug_b = slug_for_root(cfg_b.repo_root)
        assert ws.resolve(slug_b).cfg.repo_root == cfg_b.repo_root


def test_resolve_by_file_prefix(two_roots):
    cfg_a, cfg_b = two_roots
    file_inside_a = cfg_a.repo_root / "src" / "a.py"
    with Workspace([cfg_a, cfg_b]) as ws:
        assert ws.resolve(file_inside_a).cfg.repo_root == cfg_a.repo_root


def test_resolve_unknown_raises(two_roots):
    cfg_a, _cfg_b = two_roots
    with Workspace([cfg_a]) as ws:
        with pytest.raises(KeyError):
            ws.resolve("nonexistent-slug")


def test_resolve_none_returns_default(two_roots):
    cfg_a, cfg_b = two_roots
    with Workspace([cfg_a, cfg_b]) as ws:
        assert ws.resolve(None).cfg.repo_root == cfg_a.repo_root
        assert ws.resolve("").cfg.repo_root == cfg_a.repo_root


def test_list_payload_shape(two_roots):
    cfg_a, cfg_b = two_roots
    with Workspace([cfg_a, cfg_b]) as ws:
        rows = ws.list()
        assert len(rows) == 2
        assert rows[0]["default"] is True
        assert rows[1]["default"] is False
        for r in rows:
            assert {"slug", "path", "default", "watching", "last_indexed_at"} == set(r.keys())


def test_slugs_ordered(two_roots):
    cfg_a, cfg_b = two_roots
    with Workspace([cfg_a, cfg_b]) as ws:
        slugs = ws.slugs()
        assert slugs[0] == slug_for_root(cfg_a.repo_root)
        assert slugs[1] == slug_for_root(cfg_b.repo_root)


def test_writer_take_release_round_trip(two_roots):
    cfg_a, _cfg_b = two_roots
    with Workspace([cfg_a]) as ws:
        slot_before = ws.resolve(None)
        first_ro_id = id(slot_before.db_ro)
        writer = ws.take_writer(cfg_a.repo_root)
        try:
            assert writer is not None
        finally:
            ws.release_writer(cfg_a.repo_root)
        slot_after = ws.resolve(None)
        # `release_writer` reopens the read-only handle, so it must be a
        # fresh object (the caller-visible visibility quirk).
        assert id(slot_after.db_ro) != first_ro_id


def test_double_take_raises(two_roots):
    cfg_a, _cfg_b = two_roots
    with Workspace([cfg_a]) as ws:
        ws.take_writer(cfg_a.repo_root)
        try:
            with pytest.raises(RuntimeError):
                ws.take_writer(cfg_a.repo_root)
        finally:
            ws.release_writer(cfg_a.repo_root)
