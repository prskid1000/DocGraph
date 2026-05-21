from __future__ import annotations

import json
import os
from collections import deque
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
    # Cross-encoder reranker (opt-in per search call). When `rerank_default`
    # is True the API/MCP `search` endpoints default `rerank=True` so callers
    # don't have to ask for it. `rerank_model` empty = use Reranker's default
    # (jinaai/jina-reranker-v1-tiny-en).
    rerank_default: bool = False
    rerank_model: str = ""
    # Run the cross-encoder reranker on GPU (CUDA / DirectML / CoreML / ROCm
    # in that order, falling back to CPU). Off by default — the reranker model
    # is small (~33 MB) and the cost of co-located GPU inference with embed
    # passes is usually higher than the speedup. Independent from `gpu` so
    # users can keep embeddings on GPU and reranking on CPU (or vice versa).
    rerank_gpu: bool = False
    # GPU acceleration for embeddings (and reranker). Off by default; when
    # True, the Embedder asks torch for CUDA and falls back to CPU silently
    # if `torch.cuda.is_available()` is False. Requires a `+cuXY` torch
    # wheel installed; the default PyPI wheel is CPU-only.
    gpu: bool = False
    # `torch.compile` for the embedder / reranker. Off by default — costs
    # 10–30 s on first invocation but yields ~1.3–1.6× steady-state speedup
    # on GPU. Worth it for long-lived host processes; not worth it for
    # one-shot `docgraph index` runs.
    embed_torch_compile: bool = False
    rerank_torch_compile: bool = False
    workers: int = field(default_factory=lambda: max(2, (os.cpu_count() or 4) - 1))
    host: str = "127.0.0.1"
    port: int = 5500
    similar_top_k: int = 5  # SIMILAR_TO edges per node
    co_change_window: int = 200  # last N commits scanned for CO_CHANGED_WITH
    # LLM docstring augmentation (off by default — opt in via CLI flag)
    llm_docstrings: bool = False
    # LLM wiki generation. When False, build_wiki skips the LLM call and
    # renders the fact-sheet fallback even if the LLM is reachable.
    llm_wiki: bool = True
    llm_host: str = "localhost"
    llm_port: int = 1235
    llm_model: str = "qwen3.6-35b"
    llm_format: str = "openai"  # "openai" | "anthropic"
    llm_max_tokens: int = 150  # reasoning is disabled via reasoning_effort=none
    # Wiki generation needs more headroom than docstring augmentation. Kept
    # separate so users can tune the two independently — e.g. 150 for
    # docstrings, 4096+ for wiki page bodies.
    llm_max_tokens_wiki: int = 4096
    # Right-panel chat budget. 0 means "let the server decide" — for
    # OpenAI-compatible endpoints we omit max_tokens entirely so the model
    # writes until done. Anthropic format always sends a cap (its API
    # requires the field); when 0, the server falls back to its own
    # generous default (8192). Set explicitly to pin a hard limit.
    llm_max_tokens_chat: int = 0
    # API key for the LLM server (forwarded as Authorization / x-api-key
    # depending on `llm_format`). Empty = no auth header.
    llm_api_key: str = ""
    # Per-request HTTP timeout in seconds for LLM calls.
    llm_timeout: int = 1800
    # Wiki module-grouping depth used by the host's /api/wiki/build when the
    # request payload doesn't override it. 1 = top-level dirs only,
    # 12 = one page per leaf folder.
    wiki_depth: int = 12
    # Auto-unload thresholds (seconds) for the pooled embedder + reranker
    # ONNX sessions. 0 = disabled. Tuned independently so a workload that
    # embeds constantly but reranks rarely (or vice versa) can shed only
    # the truly-idle model. Mirrors telecode's `llamacpp.idle_unload_sec`
    # convention but per-model-class on the docgraph side.
    embed_unload_after: float = 0.0
    rerank_unload_after: float = 0.0
    ignore_specs: dict[Path, pathspec.PathSpec] = field(init=False)
    ignore_spec: pathspec.PathSpec = field(init=False)  # primary root, kept for back-compat

    @property
    def external_dir(self) -> Path:
        """Directory for fetched external-link HTML files: <data_dir>/external/."""
        return self.data_dir / "external"
    # Patterns the USER explicitly added (.gitignore / .docgraphignore /
    # .cursorindexingignore) — does NOT include universal generated/binary
    # defaults.
    user_ignore_specs: dict[Path, pathspec.PathSpec] = field(init=False)
    user_ignore_spec: pathspec.PathSpec = field(init=False)
    ai_block_specs: dict[Path, pathspec.PathSpec] = field(init=False)
    ai_block_spec: pathspec.PathSpec = field(init=False)
    detected_ecosystems: dict[Path, list[str]] = field(init=False)

    def __post_init__(self) -> None:
        # Auto-align embedding_dim to the chosen model. We only override when
        # the user left embedding_dim at the default (384) AND picked a model
        # whose actual dim differs — explicit dim wins so power users can
        # still override. Done first because the dim feeds Kuzu DDL.
        if self.embedding_dim == 384:
            from docgraph.embed import dim_for_model  # local import to avoid cycle
            actual = dim_for_model(self.embedding_model, default=384)
            if actual != 384:
                self.embedding_dim = actual

        # Three-tier ignore:
        #   - UNIVERSAL + autodetected ecosystem templates (docgraph.ignores)
        #   - User INDEX-EXCLUDE: .gitignore, .docgraphignore, .cursorindexingignore
        #   - AI-BLOCK (.cursorignore): file is indexed, but search/definition
        #     results are masked. Graph still includes the File node.
        self.ignore_specs = {}
        self.user_ignore_specs = {}
        self.ai_block_specs = {}
        self.detected_ecosystems = {}
        for root in [self.repo_root, *self.extra_roots]:
            index_patterns, detected = assemble_ignores(root)
            self.detected_ecosystems[root] = detected
            user_patterns: list[str] = []
            for fname in (".gitignore", ".docgraphignore", ".cursorindexingignore"):
                p = root / fname
                if p.exists():
                    user_patterns.extend(p.read_text(encoding="utf-8", errors="ignore").splitlines())
            index_patterns.extend(user_patterns)
            self.ignore_specs[root] = pathspec.PathSpec.from_lines("gitignore", index_patterns)
            self.user_ignore_specs[root] = pathspec.PathSpec.from_lines("gitignore", user_patterns)

            ai_block_patterns: list[str] = []
            ci = root / ".cursorignore"
            if ci.exists():
                ai_block_patterns.extend(ci.read_text(encoding="utf-8", errors="ignore").splitlines())
            self.ai_block_specs[root] = pathspec.PathSpec.from_lines("gitignore", ai_block_patterns)
        self.ignore_spec = self.ignore_specs[self.repo_root]
        self.user_ignore_spec = self.user_ignore_specs[self.repo_root]
        self.ai_block_spec = self.ai_block_specs[self.repo_root]

    def is_user_ignored(self, rel_path: str, root: Path | None = None) -> bool:
        """Same as is_ignored but only checks user-supplied patterns
        (.gitignore / .docgraphignore / .cursorindexingignore). Used by
        the document-indexing walker so opt-in tiers (assets like .pdf
        / .png) bypass the universal media filter while still honoring
        the user's explicit exclusions."""
        spec = self.user_ignore_specs[root] if root is not None else self.user_ignore_spec
        return spec.match_file(rel_path)

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


_DESCENT_MAX_DEPTH = 5
_DESCENT_SKIP = {"node_modules", "venv", "__pycache__", "target", "build", "dist", "out"}


def _descend_for_docgraph(start: Path) -> Path | None:
    """BFS downward from `start` looking for a `.docgraph/` directory. Returns
    the parent of the shallowest match (so launching Claude in a workspace
    folder one level above the project still works). Skips dot-folders and
    common build/cache dirs to keep the scan fast on large trees. Bounded to
    depth 5; deeper layouts should pass an explicit path."""
    queue: deque[tuple[Path, int]] = deque([(start, 0)])
    while queue:
        d, depth = queue.popleft()
        if (d / ".docgraph").is_dir():
            return d
        if depth >= _DESCENT_MAX_DEPTH:
            continue
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            name = entry.name
            if name.startswith(".") or name in _DESCENT_SKIP:
                continue
            queue.append((Path(entry.path), depth + 1))
    return None


def find_repo_root(start: Path | None = None) -> Path:
    """Resolve the repo root anchored on `.docgraph/`.

    1. Walk UP from `start` (default cwd); first ancestor with `.docgraph/`
       wins. Handles "ran from a subdirectory of an indexed project".
    2. If nothing upward, BFS DOWN from `start` (bounded depth 5) so launching
       Claude from a workspace folder one level above the project still finds
       it.
    3. Otherwise fall back to `start` itself — first `docgraph index` will
       create `.docgraph/` here.

    Note: deliberately does NOT use `.git/` as a marker. In a monorepo the
    `.git/` lives at the top while each sub-project owns its own `.docgraph/`;
    matching on `.git/` would anchor on the monorepo root and look at the
    wrong index.
    """
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".docgraph").is_dir():
            return parent
    found = _descend_for_docgraph(cur)
    if found is not None:
        return found
    return cur


def load_config(
    repo_root: Path | None = None,
    extra_roots: list[Path] | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 5500,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    embedding_dim: int = 384,
    rerank_default: bool = False,
    rerank_model: str = "",
    rerank_gpu: bool = False,
    gpu: bool = False,
    embed_torch_compile: bool = False,
    rerank_torch_compile: bool = False,
    llm_docstrings: bool = False,
    llm_wiki: bool = True,
    llm_host: str = "localhost",
    llm_port: int = 1235,
    llm_model: str = "qwen3.6-35b",
    llm_format: str = "openai",
    llm_max_tokens: int = 150,
    llm_max_tokens_wiki: int = 4096,
    llm_max_tokens_chat: int = 0,
    llm_api_key: str = "",
    llm_timeout: int = 1800,
    wiki_depth: int = 12,
    workers: int | None = None,
    embed_batch_size: int = 256,
    embed_unload_after: float = 0.0,
    rerank_unload_after: float = 0.0,
) -> Config:
    """Build a Config from explicit kwargs.

    Every knob is a parameter — no DOCGRAPH_* env vars are read. Both
    `llm_docstrings` and `llm_wiki` must be enabled explicitly (setting
    `llm_model` alone does not enable them).

    `extra_roots`, when given, overrides any persisted list and is saved
    to .docgraph/repos.json.
    """
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
        host=host,
        port=port,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        rerank_default=rerank_default,
        rerank_model=rerank_model,
        rerank_gpu=rerank_gpu,
        gpu=gpu,
        embed_torch_compile=embed_torch_compile,
        rerank_torch_compile=rerank_torch_compile,
        llm_docstrings=llm_docstrings,
        llm_wiki=llm_wiki,
        llm_host=llm_host,
        llm_port=llm_port,
        llm_model=llm_model,
        llm_format=llm_format,
        llm_max_tokens=llm_max_tokens,
        llm_max_tokens_wiki=llm_max_tokens_wiki,
        llm_max_tokens_chat=llm_max_tokens_chat,
        llm_api_key=llm_api_key,
        llm_timeout=llm_timeout,
        wiki_depth=wiki_depth,
        embed_batch_size=embed_batch_size,
        embed_unload_after=embed_unload_after,
        rerank_unload_after=rerank_unload_after,
        **({"workers": workers} if workers is not None else {}),
    )
