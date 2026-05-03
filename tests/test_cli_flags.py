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


# ── Config kwargs propagate ────────────────────────────────────────────────


@pytest.mark.parametrize("kwarg,attr,expected", [
    ("gpu", "gpu", True),
    ("embedding_model", "embedding_model", "test/model"),
    ("port", "port", 1234),
    ("host", "host", "0.0.0.0"),
    ("llm_model", "llm_model", "my-model"),
    ("llm_host", "llm_host", "remote-host"),
    ("llm_port", "llm_port", 9999),
    ("llm_format", "llm_format", "anthropic"),
    ("llm_max_tokens", "llm_max_tokens", 256),
])
def test_kwarg_propagates_to_config(tmp_path: Path,
                                     kwarg: str, attr: str, expected) -> None:
    """Every config knob is a load_config() kwarg. No DOCGRAPH_* env vars
    are read anywhere in docgraph — the host/index CLI commands forward
    flag values to load_config directly."""
    from docgraph.config import load_config
    cfg = load_config(tmp_path, **{kwarg: expected})
    actual = getattr(cfg, attr)
    if isinstance(expected, bool):
        assert bool(actual) is expected, f"{kwarg} did not flip {attr}"
    else:
        assert actual == expected, f"{kwarg} did not propagate to {attr}"


def test_llm_disabled_when_model_unset(tmp_path: Path) -> None:
    """LLM augmentation must be optional. With no `--llm-model`,
    `cfg.llm_docstrings` is False by default."""
    from docgraph.config import load_config
    cfg = load_config(tmp_path)
    assert cfg.llm_docstrings is False


def test_load_config_reads_no_env(monkeypatch, tmp_path: Path) -> None:
    """Lock in the env-free contract: setting any DOCGRAPH_* env var
    must not change the Config produced by load_config()."""
    monkeypatch.setenv("DOCGRAPH_GPU", "1")
    monkeypatch.setenv("DOCGRAPH_EMBED_MODEL", "ghost/model")
    monkeypatch.setenv("DOCGRAPH_LLM_MODEL", "ghost-llm")
    from docgraph.config import load_config
    cfg = load_config(tmp_path)
    assert cfg.gpu is False, "load_config must ignore DOCGRAPH_GPU"
    assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.llm_model == "qwen3.6-35b"


def test_llm_enabled_when_only_model_set(tmp_path: Path) -> None:
    """Setting load_config(llm_model="...") alone is enough to flip
    llm_docstrings on — same convention as `docgraph index --llm-model X`."""
    from docgraph.config import load_config
    cfg = load_config(tmp_path, llm_model="my-model")
    assert cfg.llm_docstrings is True


# ── Smoke: every CLI command can be invoked at all ─────────────────────────

@pytest.mark.parametrize("cmd", [
    "version", "index", "host", "watch", "serve", "mcp",
    "stats", "wiki", "clear", "install-mcp", "docs", "daemon",
])
def test_every_command_has_help(cmd: str) -> None:
    out = _help_text(cmd)
    assert "Usage:" in out, f"`docgraph {cmd} --help` produced no Usage line"
