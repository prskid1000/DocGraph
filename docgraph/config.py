from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

from docgraph.ignores import assemble_ignores

MAX_FILE_BYTES = 1_500_000  # 1.5 MB; skip larger files


@dataclass
class Config:
    repo_root: Path
    data_dir: Path
    db_path: Path
    cache_path: Path
    extra_roots: list[Path] = field(default_factory=list)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embed_batch_size: int = 256
    # GPU acceleration for embeddings (and reranker). Off by default; when
    # True, the Embedder asks ONNX Runtime to use CUDA / DirectML / CoreML
    # before falling back to CPU. Requires `onnxruntime-gpu` or
    # `onnxruntime-directml` to be installed.
    gpu: bool = False
    workers: int = field(default_factory=lambda: max(2, (os.cpu_count() or 4) - 1))
    host: str = "127.0.0.1"
    port: int = 5500
    similar_top_k: int = 5  # SIMILAR_TO edges per node
    co_change_window: int = 200  # last N commits scanned for CO_CHANGED_WITH
    # LLM docstring augmentation (off by default — opt in via CLI or env var)
    llm_docstrings: bool = False
    llm_host: str = "localhost"
    llm_port: int = 1235
    llm_model: str = "local-model"
    llm_format: str = "openai"  # "openai" | "anthropic"
    ignore_specs: dict[Path, pathspec.PathSpec] = field(init=False)
    ignore_spec: pathspec.PathSpec = field(init=False)  # primary root, kept for back-compat
    ai_block_specs: dict[Path, pathspec.PathSpec] = field(init=False)
    ai_block_spec: pathspec.PathSpec = field(init=False)
    detected_ecosystems: dict[Path, list[str]] = field(init=False)

    def __post_init__(self) -> None:
        # Three-tier ignore:
        #   - UNIVERSAL + autodetected ecosystem templates (docgraph.ignores)
        #   - User INDEX-EXCLUDE: .gitignore, .docgraphignore, .cursorindexingignore
        #   - AI-BLOCK (.cursorignore): file is indexed, but search/definition
        #     results are masked. Graph still includes the File node.
        self.ignore_specs = {}
        self.ai_block_specs = {}
        self.detected_ecosystems = {}
        for root in [self.repo_root, *self.extra_roots]:
            index_patterns, detected = assemble_ignores(root)
            self.detected_ecosystems[root] = detected
            for fname in (".gitignore", ".docgraphignore", ".cursorindexingignore"):
                p = root / fname
                if p.exists():
                    index_patterns.extend(p.read_text(encoding="utf-8", errors="ignore").splitlines())
            self.ignore_specs[root] = pathspec.PathSpec.from_lines("gitignore", index_patterns)

            ai_block_patterns: list[str] = []
            ci = root / ".cursorignore"
            if ci.exists():
                ai_block_patterns.extend(ci.read_text(encoding="utf-8", errors="ignore").splitlines())
            self.ai_block_specs[root] = pathspec.PathSpec.from_lines("gitignore", ai_block_patterns)
        self.ignore_spec = self.ignore_specs[self.repo_root]
        self.ai_block_spec = self.ai_block_specs[self.repo_root]

    def is_ignored(self, rel_path: str, root: Path | None = None) -> bool:
        """Should we exclude this path from indexing entirely?"""
        spec = self.ignore_specs[root] if root is not None else self.ignore_spec
        return spec.match_file(rel_path)

    def is_ai_blocked(self, rel_path: str, root: Path | None = None) -> bool:
        """Should we mask this path from AI / search results? (.cursorignore)
        Indexed but redacted — the graph still knows it exists but body and
        snippets are stripped before returning to the agent."""
        spec = self.ai_block_specs[root] if root is not None else self.ai_block_spec
        return spec.match_file(rel_path)

    def ai_blocked_logical(self, logical_rel: str) -> bool:
        """Same check, but resolves `<repo>/...` prefixed paths against the
        right root in multi-repo mode."""
        for root, prefix in self.roots_with_prefix():
            if prefix == "":
                return self.is_ai_blocked(logical_rel, root=root)
            if logical_rel.startswith(prefix):
                return self.is_ai_blocked(logical_rel[len(prefix):], root=root)
        return False

    def roots_with_prefix(self) -> list[tuple[Path, str]]:
        """Return [(absolute_root, logical_path_prefix)]. Prefix is empty when single-repo;
        otherwise it's '<basename>/' so paths are unique across repos."""
        if not self.extra_roots:
            return [(self.repo_root, "")]
        out = [(self.repo_root, self.repo_root.name + "/")]
        for r in self.extra_roots:
            out.append((r, r.name + "/"))
        return out

    def path_for(self, logical_rel: str) -> Path:
        """Map a logical (possibly prefixed) path back to its absolute filesystem location."""
        for root, prefix in self.roots_with_prefix():
            if prefix == "":
                return self.repo_root / logical_rel
            if logical_rel.startswith(prefix):
                return root / logical_rel[len(prefix):]
        return self.repo_root / logical_rel


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return cur


def load_config(
    repo_root: Path | None = None,
    extra_roots: list[Path] | None = None,
) -> Config:
    """Load config. extra_roots, when given, overrides any persisted list and is saved."""
    root = (repo_root or find_repo_root()).resolve()
    data = root / ".docgraph"
    data.mkdir(exist_ok=True)
    repos_file = data / "repos.json"

    persisted: list[Path] = []
    if repos_file.exists():
        try:
            persisted = [Path(p) for p in json.loads(repos_file.read_text())]
        except Exception:
            persisted = []

    if extra_roots is not None:
        extras = [Path(p).resolve() for p in extra_roots]
        repos_file.write_text(json.dumps([str(p) for p in extras]))
    else:
        extras = persisted

    return Config(
        repo_root=root,
        extra_roots=extras,
        data_dir=data,
        db_path=data / "graph.kuzu",
        cache_path=data / "cache.json",
        host=os.environ.get("DOCGRAPH_HOST", "127.0.0.1"),
        port=int(os.environ.get("DOCGRAPH_PORT", "5500")),
        embedding_model=os.environ.get("DOCGRAPH_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        gpu=os.environ.get("DOCGRAPH_GPU", "").lower() in ("1", "true", "yes"),
        llm_docstrings=os.environ.get("DOCGRAPH_LLM_DOCSTRINGS", "").lower() in ("1", "true", "yes"),
        llm_host=os.environ.get("DOCGRAPH_LLM_HOST", "localhost"),
        llm_port=int(os.environ.get("DOCGRAPH_LLM_PORT", "1235")),
        llm_model=os.environ.get("DOCGRAPH_LLM_MODEL", "local-model"),
        llm_format=os.environ.get("DOCGRAPH_LLM_FORMAT", "openai"),
    )
