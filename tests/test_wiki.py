"""Wiki tests: prompt/markdown formatting + resumability.

The original IndexError-on-empty-doc bug came from `splitlines()[0]` on an
empty string — easy to miss because `_wiki_prompt` was never exercised in
the suite. These cover both the formatter and the build pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from docgraph.llm import LLMClient, LLMConfig
from docgraph.wiki import (
    _facts_to_markdown,
    _wiki_prompt,
    build_wiki,
    list_wiki,
)


# --- Formatter unit tests ------------------------------------------------


def _facts_with_empty_docs() -> dict:
    """Fact sheet whose entities have no usable doc snippet — the exact
    shape that crashed _wiki_prompt before the splitlines guard."""
    return {
        "module": "src",
        "file_count": 3,
        "files": ["src/a.py", "src/b.py", "src/c.py"],
        "top_classes": [
            {"name": "A", "qname": "src/a.py::A", "file": "src/a.py", "line": 10, "doc": ""},
            {"name": "B", "qname": "src/b.py::B", "file": "src/b.py", "line": 1, "doc": None},
        ],
        "top_functions": [
            {"name": "f", "qname": "src/a.py::f", "file": "src/a.py", "line": 20, "doc": ""},
        ],
        "imports": [{"target": "json", "kind": "File"}],
        "importers": [{"file": "main.py"}],
        "tests": [{"name": "test_a", "file": "tests/test_a.py"}],
    }


def test_wiki_prompt_handles_empty_doc():
    # Pre-fix this raised IndexError from splitlines()[0] on "".
    prompt = _wiki_prompt(_facts_with_empty_docs())
    assert "src" in prompt
    assert "`A`" in prompt and "`B`" in prompt and "`f`" in prompt
    assert "## Output format" in prompt


def test_facts_to_markdown_handles_empty_doc():
    md = _facts_to_markdown(_facts_with_empty_docs())
    assert "# src" in md
    assert "**A**" in md and "**B**" in md
    assert "**f**" in md


def test_wiki_prompt_includes_present_doc():
    facts = _facts_with_empty_docs()
    facts["top_classes"][0]["doc"] = "Validates user credentials."
    prompt = _wiki_prompt(facts)
    assert "Validates user credentials." in prompt


# --- Build pipeline / resumability ---------------------------------------


class _StubLLM(LLMClient):
    """Records every prompt sent so we can assert on call count / order."""

    def __init__(self, response: str = "## Summary\nstub page body.\n") -> None:
        super().__init__(LLMConfig(model="stub"))
        self.calls: list[str] = []
        self._response = response

    def _call_openai(self, prompt: str) -> str:  # type: ignore[override]
        self.calls.append(prompt)
        return self._response

    def _call_anthropic(self, prompt: str) -> str:  # type: ignore[override]
        self.calls.append(prompt)
        return self._response


def test_wiki_build_writes_pages_and_index(indexed):
    cfg, db, _embedder, _stats = indexed
    wiki_dir = cfg.data_dir / "wiki"
    if wiki_dir.exists():
        for p in wiki_dir.iterdir():
            p.unlink()

    llm = _StubLLM()
    pages = build_wiki(cfg, db, llm)
    assert len(pages) > 0
    assert len(llm.calls) == len(pages)

    idx = json.loads((wiki_dir / "index.json").read_text(encoding="utf-8"))
    assert {e["module"] for e in idx} == {p.module for p in pages}
    for p in pages:
        body = (wiki_dir / f"{p.slug}.md").read_text(encoding="utf-8")
        assert "stub page body." in body


def test_wiki_resume_skips_existing(indexed):
    cfg, db, _embedder, _stats = indexed
    wiki_dir = cfg.data_dir / "wiki"
    for p in wiki_dir.glob("*"):
        p.unlink()

    llm1 = _StubLLM()
    pages1 = build_wiki(cfg, db, llm1)
    n_modules = len(pages1)
    assert len(llm1.calls) == n_modules

    # Second run with no force: every module already on disk → zero LLM calls.
    llm2 = _StubLLM()
    pages2 = build_wiki(cfg, db, llm2)
    assert len(pages2) == n_modules
    assert llm2.calls == [], "resume should skip modules whose .md already exists"


def test_wiki_force_rebuilds_everything(indexed):
    cfg, db, _embedder, _stats = indexed
    wiki_dir = cfg.data_dir / "wiki"
    for p in wiki_dir.glob("*"):
        p.unlink()

    llm1 = _StubLLM(response="## v1\nfirst body.\n")
    pages1 = build_wiki(cfg, db, llm1)
    n_modules = len(pages1)

    llm2 = _StubLLM(response="## v2\nsecond body.\n")
    pages2 = build_wiki(cfg, db, llm2, force=True)
    assert len(pages2) == n_modules
    assert len(llm2.calls) == n_modules
    sample = (wiki_dir / f"{pages2[0].slug}.md").read_text(encoding="utf-8")
    assert "second body." in sample


def test_wiki_resume_after_partial_crash(indexed):
    """Simulate a crash mid-run: only some modules exist on disk. Rerun
    should LLM only the missing ones, then merge into the index."""
    cfg, db, _embedder, _stats = indexed
    wiki_dir = cfg.data_dir / "wiki"
    for p in wiki_dir.glob("*"):
        p.unlink()

    # First run — full build.
    pages_full = build_wiki(cfg, db, _StubLLM())
    assert len(pages_full) >= 2, "need >=2 modules for this test"

    # Delete one module's page on disk to simulate a partial crash.
    victim = pages_full[0]
    (wiki_dir / f"{victim.slug}.md").unlink()
    (wiki_dir / f"{victim.slug}.facts.json").unlink(missing_ok=True)

    # Resume run: should LLM exactly one module, the one we deleted.
    llm = _StubLLM()
    pages = build_wiki(cfg, db, llm)
    assert len(llm.calls) == 1
    assert (wiki_dir / f"{victim.slug}.md").exists()
    # Index includes every module (none dropped during partial recovery).
    idx_modules = {e["module"] for e in list_wiki(cfg)}
    assert idx_modules == {p.module for p in pages_full}


def test_wiki_only_module_preserves_other_index_entries(indexed):
    cfg, db, _embedder, _stats = indexed
    wiki_dir = cfg.data_dir / "wiki"
    for p in wiki_dir.glob("*"):
        p.unlink()

    pages_full = build_wiki(cfg, db, _StubLLM())
    assert len(pages_full) >= 2
    target = pages_full[0].module

    llm = _StubLLM(response="## scoped\nscoped rebuild.\n")
    rebuilt = build_wiki(cfg, db, llm, only_module=target, force=True)
    assert len(rebuilt) == 1
    assert llm.calls and len(llm.calls) == 1

    idx_modules = {e["module"] for e in list_wiki(cfg)}
    assert idx_modules == {p.module for p in pages_full}, "scoped rebuild must not drop other index entries"


def test_wiki_falls_back_when_llm_returns_empty(indexed):
    """Empty LLM body → fact-sheet markdown fallback so wiki is never blank."""
    cfg, db, _embedder, _stats = indexed
    wiki_dir = cfg.data_dir / "wiki"
    for p in wiki_dir.glob("*"):
        p.unlink()

    pages = build_wiki(cfg, db, _StubLLM(response=""))
    assert len(pages) > 0
    body = (wiki_dir / f"{pages[0].slug}.md").read_text(encoding="utf-8")
    assert body.strip().startswith("# "), "fallback markdown should start with a heading"
