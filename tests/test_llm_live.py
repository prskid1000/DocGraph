"""Live LLM integration tests — opt-in.

Skipped automatically unless a local OpenAI-compatible server is reachable
at `http://localhost:1235` AND the target model (`qwen3.6-35b` by default)
is in `/v1/models`. So:

  - CI / fresh checkouts: skip (no endpoint, no model).
  - Local dev with telecode/LM Studio running and the model loaded: run.

Override the model with `DOCGRAPH_LLM_TEST_MODEL`. Tests probe the endpoint
once per session and cache the result.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

from docgraph.llm import LLMClient, LLMConfig

LIVE_HOST = os.environ.get("DOCGRAPH_LLM_TEST_HOST", "localhost")
LIVE_PORT = int(os.environ.get("DOCGRAPH_LLM_TEST_PORT", "1235"))
LIVE_MODEL = os.environ.get("DOCGRAPH_LLM_TEST_MODEL", "qwen3.6-35b")
PROBE_TIMEOUT = 2.0


def _endpoint_alive_with_model() -> tuple[bool, str]:
    """Return (ok, reason). ok=True only if the server is up AND LIVE_MODEL is loaded."""
    url = f"http://{LIVE_HOST}:{LIVE_PORT}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return False, f"endpoint {url} unreachable: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"endpoint {url} returned non-JSON: {e}"
    ids = {m.get("id") for m in (data.get("data") or [])}
    if LIVE_MODEL not in ids:
        return False, f"model {LIVE_MODEL!r} not loaded (have {sorted(ids)})"
    return True, ""


@pytest.fixture(scope="session")
def live_llm() -> LLMClient:
    ok, reason = _endpoint_alive_with_model()
    if not ok:
        pytest.skip(reason)
    return LLMClient(LLMConfig(
        host=LIVE_HOST, port=LIVE_PORT, model=LIVE_MODEL, format="openai",
    ))


def test_live_summarize_function(live_llm: LLMClient) -> None:
    """Round-trip a real call. Reasoning is disabled via reasoning_effort=none
    so a 150-token budget fits a one-sentence docstring on Qwen3 / R1."""
    body = (
        "def login(username: str, password: str) -> str:\n"
        "    if not _check_password(username, password):\n"
        "        raise TokenError('bad credentials')\n"
        "    return _issue_token(username)\n"
    )
    out = live_llm.summarize("function", "login", body, "python")
    assert out, "live LLM returned empty content (reasoning_effort=none not honored?)"
    # one short sentence — clean() truncates at first newline + 300 chars
    assert "\n" not in out
    assert len(out) <= 300
    # Sanity: should mention something authentication-shaped. Don't pin exact
    # wording — model output isn't deterministic.
    lower = out.lower()
    assert any(tok in lower for tok in ("token", "credential", "authenticat", "login", "user"))


def test_live_summarize_class(live_llm: LLMClient) -> None:
    body = (
        "class Authenticator:\n"
        "    def __init__(self, secret: str):\n"
        "        self.secret = secret\n"
        "    def login(self, u, p): ...\n"
    )
    out = live_llm.summarize("class", "Authenticator", body, "python")
    assert out
    assert "\n" not in out
