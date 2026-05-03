"""Tests for the OpenAI/Anthropic-compatible LLM client.

We stub out urllib so the tests are hermetic — no real LLM server needed.
"""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from docgraph.llm import LLMClient, LLMConfig


# --- Config -----------------------------------------------------------


def test_default_config_openai():
    cfg = LLMConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 1235
    assert cfg.format == "openai"
    assert cfg.endpoint == "http://localhost:1235/v1/chat/completions"


def test_anthropic_config_endpoint():
    cfg = LLMConfig(format="anthropic")
    assert cfg.endpoint == "http://localhost:1235/v1/messages"


def test_invalid_format_raises():
    with pytest.raises(ValueError):
        LLMConfig(format="cohere")


def test_explicit_config_kwargs():
    """LLMConfig is built from explicit kwargs — there is no env loader."""
    cfg = LLMConfig(
        port=9999, model="qwen-2.5-coder-7b",
        format="anthropic", api_key="sk-test",
    )
    assert cfg.port == 9999
    assert cfg.model == "qwen-2.5-coder-7b"
    assert cfg.format == "anthropic"
    assert cfg.api_key == "sk-test"


# --- Client (mocked HTTP) ---------------------------------------------


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        pass


def test_openai_summarize_happy():
    client = LLMClient(LLMConfig(format="openai"))
    fake_payload = {
        "choices": [
            {"message": {"content": "Authenticates a user against the database."}}
        ]
    }
    with patch("urllib.request.urlopen", return_value=_FakeResp(fake_payload)):
        out = client.summarize("function", "login", "def login(): ...", "python")
    assert out == "Authenticates a user against the database."


def test_openai_summarize_strips_quotes():
    client = LLMClient(LLMConfig(format="openai"))
    fake = {"choices": [{"message": {"content": '"Computes the answer."'}}]}
    with patch("urllib.request.urlopen", return_value=_FakeResp(fake)):
        out = client.summarize("function", "answer", "def answer(): return 42", "python")
    assert out == "Computes the answer."


def test_openai_summarize_takes_first_line():
    client = LLMClient(LLMConfig(format="openai"))
    fake = {"choices": [{"message": {"content": "Logs the user in.\n\nAlso updates the session."}}]}
    with patch("urllib.request.urlopen", return_value=_FakeResp(fake)):
        out = client.summarize("function", "login", "...", "python")
    assert out == "Logs the user in."


def test_anthropic_summarize_happy():
    client = LLMClient(LLMConfig(format="anthropic"))
    fake = {"content": [{"type": "text", "text": "Returns the user's age in years."}]}
    with patch("urllib.request.urlopen", return_value=_FakeResp(fake)):
        out = client.summarize("function", "age_of", "def age_of(u): ...", "python")
    assert out == "Returns the user's age in years."


def test_summarize_returns_empty_on_connection_error():
    """Server down → quietly returns "" (best-effort augmentation)."""
    client = LLMClient(LLMConfig(format="openai"))
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        out = client.summarize("function", "x", "def x(): ...", "python")
    assert out == ""


def test_summarize_returns_empty_on_malformed_response():
    client = LLMClient(LLMConfig(format="openai"))
    fake = {"unexpected": "shape"}
    with patch("urllib.request.urlopen", return_value=_FakeResp(fake)):
        out = client.summarize("function", "x", "def x(): ...", "python")
    assert out == ""


def test_api_key_sets_authorization_header():
    """When api_key is set, OpenAI format uses Bearer; Anthropic uses x-api-key."""
    captured: dict = {}

    def fake_urlopen(req, timeout=0):
        captured["headers"] = dict(req.headers)
        return _FakeResp({"choices": [{"message": {"content": "ok"}}]})

    client = LLMClient(LLMConfig(format="openai", api_key="sk-test"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        client.summarize("function", "x", "def x(): ...", "python")
    # urllib.request.Request lowercases header keys
    assert captured["headers"].get("Authorization") == "Bearer sk-test"

    captured.clear()

    def fake_urlopen2(req, timeout=0):
        captured["headers"] = dict(req.headers)
        return _FakeResp({"content": [{"type": "text", "text": "ok"}]})

    client_anth = LLMClient(LLMConfig(format="anthropic", api_key="sk-anthropic"))
    with patch("urllib.request.urlopen", side_effect=fake_urlopen2):
        client_anth.summarize("function", "x", "def x(): ...", "python")
    assert captured["headers"].get("X-api-key") == "sk-anthropic"
    assert "Anthropic-version" in captured["headers"]


# --- Config plumbing in Config dataclass ------------------------------


def test_config_picks_up_llm_kwargs(tmp_path):
    """Setting llm_model on load_config is enough to enable docstring
    augmentation — same convention as `docgraph index --llm-model X`."""
    from docgraph.config import load_config
    cfg = load_config(
        tmp_path,
        llm_docstrings=True,
        llm_port=11434,
        llm_model="qwen2.5-coder",
        llm_format="openai",
    )
    assert cfg.llm_docstrings is True
    assert cfg.llm_port == 11434
    assert cfg.llm_model == "qwen2.5-coder"
    assert cfg.llm_format == "openai"


def test_config_default_llm_off(tmp_path):
    from docgraph.config import load_config
    cfg = load_config(tmp_path)
    assert cfg.llm_docstrings is False
