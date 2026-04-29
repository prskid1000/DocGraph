from __future__ import annotations

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
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embed_batch_size: int = 256
    workers: int = field(default_factory=lambda: max(2, (os.cpu_count() or 4) - 1))
    host: str = "127.0.0.1"
    port: int = 5500
    similar_top_k: int = 5  # SIMILAR_TO edges per node
    co_change_window: int = 200  # last N commits scanned for CO_CHANGED_WITH
    ignore_spec: pathspec.PathSpec = field(init=False)

    def __post_init__(self) -> None:
        patterns = list(DEFAULT_IGNORES)
        gi = self.repo_root / ".gitignore"
        if gi.exists():
            patterns.extend(gi.read_text(encoding="utf-8", errors="ignore").splitlines())
        dgi = self.repo_root / ".docgraphignore"
        if dgi.exists():
            patterns.extend(dgi.read_text(encoding="utf-8", errors="ignore").splitlines())
        self.ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def is_ignored(self, rel_path: str) -> bool:
        return self.ignore_spec.match_file(rel_path)


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return cur


def load_config(repo_root: Path | None = None) -> Config:
    root = (repo_root or find_repo_root()).resolve()
    data = root / ".docgraph"
    data.mkdir(exist_ok=True)
    return Config(
        repo_root=root,
        data_dir=data,
        db_path=data / "graph.kuzu",
        cache_path=data / "cache.json",
        host=os.environ.get("DOCGRAPH_HOST", "127.0.0.1"),
        port=int(os.environ.get("DOCGRAPH_PORT", "5500")),
        embedding_model=os.environ.get("DOCGRAPH_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
    )
