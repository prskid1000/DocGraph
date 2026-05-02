"""Lock in every CLI flag and env var that telecode passes.

If telecode's `docgraph/process.py` invokes `docgraph index --workers 4`
and docgraph's CLI doesn't accept `--workers`, that's a silent breakage
in the index runner. These tests make any future flag rename break a
test instead.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DOCGRAPH = shutil.which("docgraph") or str(Path(__file__).parent.parent / ".venv" / "Scripts" / "docgraph.exe")


def _help_text(*args: str) -> str:
    """Return `docgraph <args...> --help` stdout+stderr as a single string."""
    proc = subprocess.run(
        [DOCGRAPH, *args, "--help"],
        capture_output=True, text=True, timeout=20,
        env={**os.environ, "FORCE_COLOR": "0", "NO_COLOR": "1"},
    )
    return proc.stdout + proc.stderr


# ── Flags telecode passes to `docgraph index` ──────────────────────────────


@pytest.fixture(scope="module")
def index_help() -> str:
    return _help_text("index")


@pytest.mark.parametrize("flag", [
    "--full",
    "--workers",
    "--gpu",
    "--llm-model",
    "--llm-host",
    "--llm-port",
    "--llm-format",
    "--llm-max-tokens",
])
def test_index_accepts_telecode_flag(index_help: str, flag: str) -> None:
    assert flag in index_help, f"docgraph index missing {flag}"


# ── Flags telecode passes to `docgraph host` ───────────────────────────────


@pytest.fixture(scope="module")
def host_help() -> str:
    return _help_text("host")


@pytest.mark.parametrize("flag", ["--root", "--watch", "--host", "--port"])
def test_host_accepts_telecode_flag(host_help: str, flag: str) -> None:
    assert flag in host_help, f"docgraph host missing {flag}"


# ── Env vars docgraph reads ────────────────────────────────────────────────


@pytest.mark.parametrize("var,attr,expected", [
    ("DOCGRAPH_GPU", "gpu", True),
    ("DOCGRAPH_EMBED_MODEL", "embedding_model", "test/model"),
    ("DOCGRAPH_PORT", "port", 1234),
    ("DOCGRAPH_HOST", "host", "0.0.0.0"),
    ("DOCGRAPH_LLM_MODEL", "llm_model", "my-model"),
    ("DOCGRAPH_LLM_HOST", "llm_host", "remote-host"),
    ("DOCGRAPH_LLM_PORT", "llm_port", 9999),
    ("DOCGRAPH_LLM_FORMAT", "llm_format", "anthropic"),
    ("DOCGRAPH_LLM_MAX_TOKENS", "llm_max_tokens", 256),
])
def test_env_var_propagates_to_config(monkeypatch, tmp_path: Path,
                                       var: str, attr: str, expected) -> None:
    """telecode controls docgraph host behavior partly via env vars
    (DOCGRAPH_GPU, DOCGRAPH_EMBED_MODEL). Verify every env var the
    config layer reads still propagates to the resulting Config object."""
    # Clear all DOCGRAPH_* vars first so the test isn't polluted by host env.
    for key in list(os.environ):
        if key.startswith("DOCGRAPH_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(var, str(expected))
    from docgraph.config import load_config
    cfg = load_config(tmp_path)
    actual = getattr(cfg, attr)
    if isinstance(expected, bool):
        assert bool(actual) is expected, f"{var} did not flip {attr}"
    else:
        assert actual == expected, f"{var} did not propagate to {attr}"


def test_llm_disabled_when_model_unset(monkeypatch, tmp_path: Path) -> None:
    """LLM augmentation must be optional. With no DOCGRAPH_LLM_* vars set
    and no --llm-model on the CLI, `cfg.llm_docstrings` must be False."""
    for key in list(os.environ):
        if key.startswith("DOCGRAPH_LLM_"):
            monkeypatch.delenv(key, raising=False)
    from docgraph.config import load_config
    cfg = load_config(tmp_path)
    assert cfg.llm_docstrings is False, (
        "LLM augmentation should default off when no DOCGRAPH_LLM_* vars are set"
    )


def test_llm_enabled_when_only_model_set(monkeypatch, tmp_path: Path) -> None:
    """Setting DOCGRAPH_LLM_MODEL alone is enough to flip llm_docstrings on
    (the docstring contract from config.py)."""
    for key in list(os.environ):
        if key.startswith("DOCGRAPH_LLM_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DOCGRAPH_LLM_MODEL", "my-model")
    from docgraph.config import load_config
    cfg = load_config(tmp_path)
    assert cfg.llm_docstrings is True


# ── Smoke: every CLI command can be invoked at all ─────────────────────────

@pytest.mark.parametrize("cmd", [
    "version", "index", "host", "watch", "serve", "mcp",
    "stats", "wiki", "clear", "install-mcp", "docs", "daemon",
])
def test_every_command_has_help(cmd: str) -> None:
    out = _help_text(cmd)
    assert "Usage:" in out, f"`docgraph {cmd} --help` produced no Usage line"
