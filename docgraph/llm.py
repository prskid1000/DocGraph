"""Tiny LLM client for opt-in docstring generation.

Talks to a local OpenAI- or Anthropic-compatible Chat endpoint. Defaults
to `http://localhost:1235` with OpenAI Chat Completions format — plays
nicely with LM Studio, llama.cpp, vLLM, Ollama-with-OpenAI-compat.

Configurable via CLI flags or env vars (no settings file):

    --llm-docstrings        / DOCGRAPH_LLM_DOCSTRINGS=1
    --llm-port    <int>     / DOCGRAPH_LLM_PORT (default: 1235)
    --llm-model   <str>     / DOCGRAPH_LLM_MODEL (default: "qwen3.6-35b")
    --llm-format  openai|anthropic / DOCGRAPH_LLM_FORMAT (default: "openai")
    (api key)               / DOCGRAPH_LLM_API_KEY (optional)

Stdlib urllib only — no extra deps. Off by default; the indexer skips
this entire step unless `cfg.llm_docstrings` is true.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_PORT = 1235
DEFAULT_MODEL = "qwen3.6-35b"
DEFAULT_FORMAT = "openai"
# We send `reasoning_effort: "none"` on every call (see _call_openai), which
# the telecode proxy resolves into the right "no thinking" knobs per model
# family (enable_thinking=false / thinking_budget_tokens=0 for Qwen3, etc.)
# Without that flag, reasoning models eat 500-2000 tokens before the answer
# and a 150-token budget comes back with empty content. The flag makes 150
# enough for a one-sentence docstring; bump it via env / CLI if needed.
DEFAULT_MAX_TOKENS = 150
DEFAULT_TIMEOUT_SECS = 60


_PROMPT = (
    "Write a single-sentence docstring (under 25 words) for this {kind} "
    "named `{name}` in {language}. Describe its purpose, not its implementation. "
    "Return only the sentence — no quotes, no markdown, no preamble.\n\n"
    "```{language}\n{body}\n```"
)


@dataclass
class LLMConfig:
    host: str = "localhost"
    port: int = DEFAULT_PORT
    model: str = DEFAULT_MODEL
    format: str = DEFAULT_FORMAT  # "openai" or "anthropic"
    api_key: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = DEFAULT_TIMEOUT_SECS

    def __post_init__(self) -> None:
        fmt = (self.format or "").lower()
        if fmt not in ("openai", "anthropic"):
            raise ValueError(f"llm format must be 'openai' or 'anthropic', got {self.format!r}")
        self.format = fmt

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def endpoint(self) -> str:
        if self.format == "anthropic":
            return f"{self.base_url}/messages"
        return f"{self.base_url}/chat/completions"


class LLMClient:
    def __init__(self, cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg or LLMConfig()

    def summarize(self, kind: str, name: str, body: str, language: str) -> str:
        """Generate a one-sentence docstring. Returns "" on any failure —
        callers treat the LLM as best-effort augmentation."""
        snippet = body[:3000]
        prompt = _PROMPT.format(kind=kind, name=name, language=language, body=snippet)
        try:
            if self.cfg.format == "anthropic":
                return self._call_anthropic(prompt)
            return self._call_openai(prompt)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log.debug(f"LLM call failed for {name}: {e}")
            return ""
        except Exception as e:  # noqa: BLE001
            log.debug(f"LLM unexpected error for {name}: {e}")
            return ""

    def _call_openai(self, prompt: str) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0.2,
            "stream": False,
            # Disable chain-of-thought on reasoning models. The telecode proxy
            # at :1235 maps this to per-model knobs (Qwen3: enable_thinking=
            # false + thinking_budget_tokens=0; DeepSeek-R1: similar). Plain
            # OpenAI / non-reasoning servers ignore the field. Without this
            # flag a 150-token budget comes back with empty `content` because
            # reasoning consumed it all.
            "reasoning_effort": "none",
        }
        data = self._post(self.cfg.endpoint, payload)
        try:
            return self._clean(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            return ""

    def _call_anthropic(self, prompt: str) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0.2,
        }
        data = self._post(self.cfg.endpoint, payload)
        try:
            for b in data.get("content", []):
                if b.get("type") == "text":
                    return self._clean(b.get("text", ""))
        except (AttributeError, TypeError):
            pass
        return ""

    def _post(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            if self.cfg.format == "anthropic":
                headers["x-api-key"] = self.cfg.api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _clean(text: str) -> str:
        t = (text or "").strip()
        for ch in ('"', "'", "`"):
            if t.startswith(ch) and t.endswith(ch) and len(t) > 2:
                t = t[1:-1].strip()
        first = t.splitlines()[0].strip() if t else ""
        return first[:300]


# --- Env helpers ------------------------------------------------------


def llm_config_from_env() -> LLMConfig:
    """Build LLMConfig from DOCGRAPH_LLM_* env vars. Used as the baseline
    that CLI flags override. Always returns a valid config — callers
    decide whether to actually call out based on `cfg.llm_docstrings`."""
    import os
    return LLMConfig(
        host=os.environ.get("DOCGRAPH_LLM_HOST", "localhost"),
        port=int(os.environ.get("DOCGRAPH_LLM_PORT", DEFAULT_PORT)),
        model=os.environ.get("DOCGRAPH_LLM_MODEL", DEFAULT_MODEL),
        format=os.environ.get("DOCGRAPH_LLM_FORMAT", DEFAULT_FORMAT),
        api_key=os.environ.get("DOCGRAPH_LLM_API_KEY") or None,
        max_tokens=int(os.environ.get("DOCGRAPH_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        timeout=int(os.environ.get("DOCGRAPH_LLM_TIMEOUT", DEFAULT_TIMEOUT_SECS)),
    )
