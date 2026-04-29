"""Unit tests that don't need a built index — config, summary, watch filter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from docgraph.config import Config, load_config
from docgraph.summary import (
    build_embedding_text,
    extract_docstring,
    smart_body_sample,
)


# --- summary.extract_docstring ----------------------------------------


def test_extract_python_docstring():
    body = '''def foo(x):
    """Add one to x."""
    return x + 1
'''
    doc = extract_docstring(body, "python")
    assert "Add one" in doc


def test_extract_python_triple_single_quotes():
    body = "def foo():\n    '''single quoted doc'''\n    pass\n"
    doc = extract_docstring(body, "python")
    assert "single quoted" in doc


def test_extract_jsdoc():
    body = """/**
 * Logs the user in.
 * @param u username
 */
function login(u) { return u; }
"""
    doc = extract_docstring(body, "javascript")
    assert "Logs the user in" in doc


def test_extract_rust_doc_lines():
    body = """/// Compute the answer.
/// Returns 42.
fn answer() -> i32 { 42 }
"""
    doc = extract_docstring(body, "rust")
    assert "Compute the answer" in doc


def test_extract_no_docstring():
    body = "def foo(x):\n    return x\n"
    assert extract_docstring(body, "python") == ""


# --- summary.smart_body_sample ----------------------------------------


def test_smart_body_short_unchanged():
    body = "short body"
    assert smart_body_sample(body) == body


def test_smart_body_long_takes_head_and_tail():
    body = "A" * 2000 + "MIDDLE" + "B" * 2000
    s = smart_body_sample(body)
    assert s.startswith("A")
    assert s.endswith("B")
    assert "MIDDLE" not in s
    assert "…" in s  # marker between head and tail


# --- summary.build_embedding_text -------------------------------------


def test_build_embedding_text_includes_name_and_doc():
    text = build_embedding_text(
        name="login",
        qname="auth.py::Authenticator::login",
        signature="def login(self, username, password) -> str",
        body='def login(self, u, p):\n    """Verify and return token."""\n    return "x"',
        language="python",
        kind="method",
    )
    assert "login" in text
    assert "Verify and return token" in text
    assert "method login" in text  # kind prefix


def test_build_embedding_text_truncated():
    body = "x" * 50000
    text = build_embedding_text("f", "f", "", body, "python", "function")
    # MAX_EMBED_CHARS = 2200
    assert len(text) <= 2200


# --- Config + load_config persistence --------------------------------


def test_load_config_creates_data_dir(tmp_path: Path):
    cfg = load_config(tmp_path)
    assert cfg.data_dir.exists()
    assert cfg.db_path.parent == cfg.data_dir


def test_load_config_persists_extra_roots(tmp_path: Path):
    other = tmp_path.parent / "other_repo"
    other.mkdir(exist_ok=True)
    cfg = load_config(tmp_path, extra_roots=[other])
    repos_file = cfg.data_dir / "repos.json"
    assert repos_file.exists()
    data = json.loads(repos_file.read_text())
    assert any(str(other) == p or str(other.resolve()) == p for p in data)
    # Reload without args should pick up persisted list
    cfg2 = load_config(tmp_path)
    assert any(str(p).endswith("other_repo") for p in cfg2.extra_roots)


def test_roots_with_prefix_single(tmp_path: Path):
    cfg = load_config(tmp_path)
    roots = cfg.roots_with_prefix()
    assert len(roots) == 1
    assert roots[0][1] == ""


def test_roots_with_prefix_multi(tmp_path: Path):
    other = tmp_path.parent / "second"
    other.mkdir(exist_ok=True)
    cfg = load_config(tmp_path, extra_roots=[other])
    roots = cfg.roots_with_prefix()
    assert len(roots) == 2
    # Prefix is "<basename>/"
    assert all(p.endswith("/") for _, p in roots)


def test_path_for_roundtrip(tmp_path: Path):
    cfg = load_config(tmp_path)
    p = cfg.path_for("src/foo.py")
    assert p == tmp_path / "src" / "foo.py"


# --- Watch filter ------------------------------------------------------


def test_watch_filter_ignores_non_repo(tmp_path: Path):
    from docgraph.watch import _is_relevant
    cfg = load_config(tmp_path)
    # Path outside repo
    assert _is_relevant(cfg, Path("/totally/elsewhere/file.py")) is False


def test_watch_filter_accepts_python(tmp_path: Path):
    from docgraph.watch import _is_relevant
    cfg = load_config(tmp_path)
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("x = 1\n")
    assert _is_relevant(cfg, src) is True


def test_watch_filter_rejects_unsupported_extension(tmp_path: Path):
    from docgraph.watch import _is_relevant
    cfg = load_config(tmp_path)
    p = tmp_path / "notes.md"
    p.write_text("hi")
    # Markdown isn't in EXT_TO_LANG
    assert _is_relevant(cfg, p) is False


def test_watch_filter_respects_ignore(tmp_path: Path):
    from docgraph.watch import _is_relevant
    (tmp_path / ".docgraphignore").write_text("ignored/\n")
    cfg = load_config(tmp_path)
    p = tmp_path / "ignored" / "x.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x=1")
    assert _is_relevant(cfg, p) is False
