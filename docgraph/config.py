from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

DEFAULT_IGNORES = [
    ".git/",
    ".docgraph/",
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    "dist/",
    "build/",
    "target/",
    ".next/",
    ".cache/",
    "*.min.js",
    "*.min.css",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "*.svg",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.pdf",
    "*.zip",
    "*.tar*",
    "*.log",
]

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
    workers: int = field(default_factory=lambda: max(2, (os.cpu_count() or 4) - 1))
    host: str = "127.0.0.1"
    port: int = 5500
    similar_top_k: int = 5  # SIMILAR_TO edges per node
    co_change_window: int = 200  # last N commits scanned for CO_CHANGED_WITH
    ignore_specs: dict[Path, pathspec.PathSpec] = field(init=False)
    ignore_spec: pathspec.PathSpec = field(init=False)  # primary root, kept for back-compat

    def __post_init__(self) -> None:
        self.ignore_specs = {}
        for root in [self.repo_root, *self.extra_roots]:
            patterns = list(DEFAULT_IGNORES)
            gi = root / ".gitignore"
            if gi.exists():
                patterns.extend(gi.read_text(encoding="utf-8", errors="ignore").splitlines())
            dgi = root / ".docgraphignore"
            if dgi.exists():
                patterns.extend(dgi.read_text(encoding="utf-8", errors="ignore").splitlines())
            self.ignore_specs[root] = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        self.ignore_spec = self.ignore_specs[self.repo_root]

    def is_ignored(self, rel_path: str, root: Path | None = None) -> bool:
        spec = self.ignore_specs[root] if root is not None else self.ignore_spec
        return spec.match_file(rel_path)

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
    )
