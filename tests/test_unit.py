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


# --- Ecosystem autodetect (docgraph.ignores) --------------------------


def test_universal_ignores_applied_without_markers(tmp_path: Path):
    cfg = load_config(tmp_path)
    assert cfg.detected_ecosystems[tmp_path.resolve()] == []
    # Universal patterns still active
    assert cfg.is_ignored(".git/HEAD")
    assert cfg.is_ignored("foo.png")
    assert cfg.is_ignored("yarn.lock")
    assert cfg.is_ignored(".env")
    assert cfg.is_ignored(".env.production")
    assert cfg.is_ignored(".idea/workspace.xml")
    assert cfg.is_ignored(".DS_Store")


def test_node_autodetect(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"name": "x"}')
    cfg = load_config(tmp_path)
    assert "node" in cfg.detected_ecosystems[tmp_path.resolve()]
    assert cfg.is_ignored("node_modules/foo/bar.js")
    assert cfg.is_ignored(".next/cache/foo")
    assert cfg.is_ignored(".turbo/run.json")
    assert cfg.is_ignored("dist/bundle.js")


def test_angular_autodetect_pulls_node_too(tmp_path: Path):
    (tmp_path / "angular.json").write_text("{}")
    (tmp_path / "package.json").write_text('{"name": "ng"}')
    cfg = load_config(tmp_path)
    eco = cfg.detected_ecosystems[tmp_path.resolve()]
    assert "angular" in eco
    assert "node" in eco
    assert cfg.is_ignored(".angular/cache/foo")
    assert cfg.is_ignored("out-tsc/foo.js")
    assert cfg.is_ignored("node_modules/x")


def test_python_autodetect(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    cfg = load_config(tmp_path)
    assert "python" in cfg.detected_ecosystems[tmp_path.resolve()]
    assert cfg.is_ignored("__pycache__/foo.pyc")
    assert cfg.is_ignored(".venv/lib/x.py")
    assert cfg.is_ignored(".ruff_cache/foo")
    assert cfg.is_ignored(".mypy_cache/foo")


def test_maven_java_autodetect(tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project/>")
    cfg = load_config(tmp_path)
    eco = cfg.detected_ecosystems[tmp_path.resolve()]
    assert "maven" in eco
    assert "java" in eco
    assert cfg.is_ignored("target/classes/Foo.class")
    assert cfg.is_ignored(".mvn/wrapper/maven-wrapper.jar")
    assert cfg.is_ignored("Main.class")


def test_gradle_autodetect_picks_build(tmp_path: Path):
    (tmp_path / "build.gradle").write_text("")
    cfg = load_config(tmp_path)
    eco = cfg.detected_ecosystems[tmp_path.resolve()]
    assert "gradle" in eco
    assert "java" in eco
    assert cfg.is_ignored(".gradle/caches/foo")
    assert cfg.is_ignored("app/build/output.jar")


def test_rust_autodetect(tmp_path: Path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    cfg = load_config(tmp_path)
    assert "rust" in cfg.detected_ecosystems[tmp_path.resolve()]
    assert cfg.is_ignored("target/debug/foo")


def test_dotnet_glob_marker(tmp_path: Path):
    (tmp_path / "App.csproj").write_text("<Project/>")
    cfg = load_config(tmp_path)
    assert "dotnet" in cfg.detected_ecosystems[tmp_path.resolve()]
    assert cfg.is_ignored("bin/Debug/App.dll")
    assert cfg.is_ignored("obj/Debug/foo")


def test_user_gitignore_layered_on_top(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"name":"x"}')
    (tmp_path / ".gitignore").write_text("custom_dir/\n")
    cfg = load_config(tmp_path)
    assert cfg.is_ignored("node_modules/foo")  # autodetect
    assert cfg.is_ignored("custom_dir/x.js")    # user .gitignore


def test_universal_covers_data_science_and_misc(tmp_path: Path):
    """Jupyter / MLflow / wandb / R / Haskell / Zig / Scala-tooling — all
    universal because the dir names are unambiguous."""
    cfg = load_config(tmp_path)
    assert cfg.is_ignored(".ipynb_checkpoints/Untitled.ipynb")
    assert cfg.is_ignored("mlruns/0/abc/meta.yaml")
    assert cfg.is_ignored("wandb/run-20260101/files/foo")
    assert cfg.is_ignored("lightning_logs/version_0/events.out")
    assert cfg.is_ignored(".dvc/cache/00/abc")
    assert cfg.is_ignored(".Rproj.user/foo")
    assert cfg.is_ignored(".Rhistory")
    assert cfg.is_ignored(".stack-work/install/foo")
    assert cfg.is_ignored("dist-newstyle/build/foo")
    assert cfg.is_ignored("zig-cache/o/abc")
    assert cfg.is_ignored("zig-out/bin/app")
    assert cfg.is_ignored(".metals/readonly")
    assert cfg.is_ignored(".bloop/foo")


def test_scala_autodetect(tmp_path: Path):
    (tmp_path / "build.sbt").write_text('name := "x"\n')
    cfg = load_config(tmp_path)
    assert "scala" in cfg.detected_ecosystems[tmp_path.resolve()]
    assert cfg.is_ignored("target/scala-2.13/foo.class")
    assert cfg.is_ignored("project/target/foo")


def test_unknown_repo_universal_only(tmp_path: Path):
    """Plain text repo with no markers: still gets universal patterns
    (unambiguously-named dep dirs + binaries), but NOT ambiguous build dirs."""
    (tmp_path / "README.txt").write_text("hi")
    cfg = load_config(tmp_path)
    assert cfg.detected_ecosystems[tmp_path.resolve()] == []
    # Unambiguous dep dirs always ignored (Cursor parity)
    assert cfg.is_ignored("node_modules/foo")
    assert cfg.is_ignored("__pycache__/foo")
    assert cfg.is_ignored(".venv/lib/x.py")
    assert cfg.is_ignored(".gradle/caches/foo")
    # Binaries always ignored
    assert cfg.is_ignored("logo.png")
    # Ambiguous build dirs NOT ignored without ecosystem detection
    assert not cfg.is_ignored("target/foo.txt")
    assert not cfg.is_ignored("bin/foo.txt")
    assert not cfg.is_ignored("obj/foo.txt")
