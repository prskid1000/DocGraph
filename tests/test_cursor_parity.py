"""Tests for Cursor-parity features:
- git_changes / git_blame / git_recent
- rules_for(file) (.cursor/rules/*.mdc + AGENTS.md)
- Two-tier ignore: .cursorignore (AI-block) vs .cursorindexingignore (skip)
- Sub-function chunking + max-pooled search
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.git_tools import changed_line_ranges
from docgraph.index import Indexer
from docgraph.retrieve import Retriever
from docgraph.rules import _parse_frontmatter, collect_rules, rules_for


# --- diff parser -------------------------------------------------------


def test_changed_line_ranges_single_hunk():
    diff = "@@ -1,3 +1,5 @@\n hello\n+new line\n+another\n world\n"
    assert changed_line_ranges(diff) == [(1, 5)]


def test_changed_line_ranges_multi_hunk():
    diff = (
        "@@ -10,2 +10,3 @@\n line\n+added\n line\n"
        "@@ -20,1 +21,2 @@\n other\n+added\n"
    )
    assert changed_line_ranges(diff) == [(10, 12), (21, 22)]


def test_changed_line_ranges_no_count():
    # +N alone (without ,M) means a single line
    diff = "@@ -5 +5 @@\n-old\n+new\n"
    assert changed_line_ranges(diff) == [(5, 5)]


# --- frontmatter parser ------------------------------------------------


def test_parse_frontmatter_basic():
    text = "---\ndescription: hi\nglobs: ['*.py']\nalwaysApply: true\n---\nbody here"
    fm, body = _parse_frontmatter(text)
    assert fm["description"] == "hi"
    assert fm["globs"] == ["*.py"]
    assert fm["alwaysApply"] is True
    assert body.strip() == "body here"


def test_parse_frontmatter_no_frontmatter():
    fm, body = _parse_frontmatter("just a body\nno frontmatter")
    assert fm == {}
    assert body == "just a body\nno frontmatter"


def test_parse_frontmatter_unquoted_globs_list():
    text = "---\nglobs: [src/**/*.ts, tests/**/*.spec.ts]\n---\nx"
    fm, _ = _parse_frontmatter(text)
    assert fm["globs"] == ["src/**/*.ts", "tests/**/*.spec.ts"]


# --- rules_for ---------------------------------------------------------


@pytest.fixture
def repo_with_rules(tmp_path: Path) -> Path:
    (tmp_path / ".cursor" / "rules").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".cursor" / "rules" / "ts-tests.mdc").write_text(
        "---\ndescription: TypeScript test rules\nglobs: ['**/*.test.ts', '**/*.spec.ts']\n---\nUse Vitest. Mock with vi.fn().\n",
        encoding="utf-8",
    )
    (tmp_path / ".cursor" / "rules" / "always.mdc").write_text(
        "---\ndescription: Project-wide\nalwaysApply: true\n---\nNever use any.\n",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text("# Agent guide\n\nBe terse.\n", encoding="utf-8")
    return tmp_path


def test_collect_rules_finds_all(repo_with_rules: Path):
    cfg = load_config(repo_with_rules)
    rules = collect_rules(cfg)
    names = {r.name for r in rules}
    assert "ts-tests" in names
    assert "always" in names
    assert "AGENTS.md" in names


def test_rules_for_matches_globs(repo_with_rules: Path):
    cfg = load_config(repo_with_rules)
    matched = rules_for(cfg, "src/foo.test.ts")
    names = {r["name"] for r in matched}
    assert "ts-tests" in names
    # always-apply rules + AGENTS.md should also appear
    assert "always" in names
    assert "AGENTS.md" in names


def test_rules_for_no_glob_match(repo_with_rules: Path):
    cfg = load_config(repo_with_rules)
    matched = rules_for(cfg, "src/foo.py")
    names = {r["name"] for r in matched}
    # Glob-specific rule should NOT match
    assert "ts-tests" not in names
    # Always-apply rules should
    assert "always" in names
    assert "AGENTS.md" in names


def test_rules_for_always_apply_first(repo_with_rules: Path):
    cfg = load_config(repo_with_rules)
    matched = rules_for(cfg, "src/foo.test.ts")
    # Always-apply rules come before glob-specific rules
    always_idx = next(i for i, r in enumerate(matched) if r["always_apply"])
    glob_idx = next((i for i, r in enumerate(matched) if r["name"] == "ts-tests"), None)
    assert glob_idx is None or always_idx < glob_idx


# --- Two-tier ignore ---------------------------------------------------


def test_cursorindexingignore_excludes_from_walk(tmp_path: Path):
    (tmp_path / ".cursorindexingignore").write_text("vendor/\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    assert cfg.is_ignored("vendor/foo.py") is True
    # Things outside the ignore stay walkable
    assert cfg.is_ignored("src/foo.py") is False


def test_cursorignore_blocks_ai_but_not_index(tmp_path: Path):
    (tmp_path / ".cursorignore").write_text("secrets.py\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    assert cfg.is_ai_blocked("secrets.py") is True
    # Crucially: still indexable (not in ignore_spec)
    assert cfg.is_ignored("secrets.py") is False


def test_ai_blocked_logical_multi_repo(tmp_path: Path):
    other = tmp_path.parent / "other"
    other.mkdir(exist_ok=True)
    (other / ".cursorignore").write_text("private.py\n", encoding="utf-8")
    cfg = load_config(tmp_path, extra_roots=[other])
    # Logical path includes the repo prefix
    rel = f"{other.name}/private.py"
    assert cfg.ai_blocked_logical(rel) is True
    # Sibling unblocked path
    assert cfg.ai_blocked_logical(f"{other.name}/public.py") is False


# --- Sub-function chunking --------------------------------------------


def test_chunks_created_for_long_function(indexed):
    _cfg, db, _e, _s = indexed
    rows = db.fetch_all("MATCH (c:Chunk) RETURN c.parent_qname AS q, c.idx AS idx")
    # process_pipeline should be chunked (its body is well over 1500 chars)
    parents = {r["q"] for r in rows}
    assert any("process_pipeline" in q for q in parents), (
        f"expected process_pipeline to be chunked; got {parents}"
    )


def test_chunks_have_embeddings(indexed):
    _cfg, db, _e, _s = indexed
    rows = db.fetch_all("MATCH (c:Chunk) RETURN c.embedding AS e LIMIT 5")
    if not rows:
        pytest.skip("no chunks; threshold may have changed")
    # First element of an embedding is a float (not all zeros)
    first = rows[0]["e"]
    assert len(first) == 384
    assert any(abs(x) > 1e-6 for x in first)


def test_chunked_search_recall(retriever):
    # Search for something only mentioned inside one stage of process_pipeline
    # ("normalize"). Without chunking we'd dilute the match across the whole
    # 1500-char body; with chunking, the per-stage chunk wins.
    results = retriever.search("normalize records lowercase", limit=10)
    names = [r["name"] for r in results]
    assert "process_pipeline" in names


# --- git_changes / git_blame / git_recent ----------------------------


def _make_git_repo(root: Path) -> None:
    (root / "a.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=False)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=False)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=False)
    subprocess.run(["git", "add", "."], cwd=root, check=False)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=False)


def test_git_recent_returns_commits(tmp_path: Path):
    _make_git_repo(tmp_path)
    cfg = load_config(tmp_path)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    r = Retriever(db, embedder, cfg=cfg)
    commits = r.git_recent(limit=5)
    assert commits
    assert commits[0]["subject"] == "init"


def test_git_changes_returns_files_for_dirty_tree(tmp_path: Path):
    _make_git_repo(tmp_path)
    # Modify a.py without committing
    (tmp_path / "a.py").write_text("def hello():\n    return 2  # changed\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    r = Retriever(db, embedder, cfg=cfg)
    out = r.git_changes(ref=None)  # working tree
    assert any("a.py" in f["path"] for f in out["files"])


def test_git_blame_lines(tmp_path: Path):
    _make_git_repo(tmp_path)
    cfg = load_config(tmp_path)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    r = Retriever(db, embedder, cfg=cfg)
    rows = r.git_blame("a.py", line_start=1, line_end=2)
    assert rows
    assert all("commit" in row and "author" in row for row in rows)
