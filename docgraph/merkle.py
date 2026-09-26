"""File-hash tree for fast startup scans.

The delta indexer used to SHA1 every file on every run -- reading the whole
repo off disk just to learn that nothing changed. This keeps, per file,
`(size, mtime_ns, sha1)` and, per directory, a Merkle hash over its
children, in `.docgraph/merkle.json` (a cache next to cache.json, not a
second store -- losing it only costs one full hashing pass).

scan():
  * a file whose (size, mtime_ns) matches the previous scan reuses its
    stored hash without being read;
  * everything else is read + hashed;
  * directory hashes are recomputed bottom-up, and `unchanged_dirs` lists
    the directories whose hash matches the previous tree -- whole subtrees
    the caller can skip when diffing.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
from dataclasses import dataclass, field
from pathlib import Path

VERSION = 1


@dataclass
class ScanResult:
    hashes: dict[str, str]                       # logical rel -> sha1
    dir_hashes: dict[str, str]
    root_hash: str
    hashed: int = 0                              # files actually read
    reused: int = 0                              # files whose stat matched
    unchanged_dirs: set[str] = field(default_factory=set)
    root_unchanged: bool = False


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != VERSION:
            return {}
        return data
    except Exception:
        return {}


def save(path: Path, result: ScanResult, stats: dict[str, tuple[int, int]]) -> None:
    files = {rel: [stats[rel][0], stats[rel][1], h]
             for rel, h in result.hashes.items() if rel in stats}
    payload = {"version": VERSION, "root": result.root_hash,
               "dirs": result.dir_hashes, "files": files}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def scan(files: list[tuple[Path, str]], previous: dict,
         force_rehash: bool = False) -> tuple[ScanResult, dict[str, tuple[int, int]]]:
    """files: [(absolute_path, logical_rel)] from the walker."""
    prev_files: dict = previous.get("files", {}) if previous else {}
    prev_dirs: dict = previous.get("dirs", {}) if previous else {}
    hashes: dict[str, str] = {}
    stats: dict[str, tuple[int, int]] = {}
    hashed = reused = 0
    for path, rel in files:
        try:
            st = path.stat()
            key = (int(st.st_size), int(st.st_mtime_ns))
        except OSError:
            continue
        stats[rel] = key
        old = prev_files.get(rel)
        if (not force_rehash and old and len(old) == 3
                and int(old[0]) == key[0] and int(old[1]) == key[1] and old[2]):
            hashes[rel] = old[2]
            reused += 1
        else:
            hashes[rel] = _sha1_file(path)
            hashed += 1

    # Bottom-up directory hashes: dir -> sorted children (name, hash).
    children: dict[str, list[tuple[str, str]]] = {}
    for rel, h in hashes.items():
        d = posixpath.dirname(rel)
        children.setdefault(d, []).append((posixpath.basename(rel), h))
        # make sure every ancestor exists
        while d:
            parent = posixpath.dirname(d)
            children.setdefault(parent, [])
            d = parent
    dir_hashes: dict[str, str] = {}
    subdirs: dict[str, list[str]] = {}
    for d in children:
        if d:
            subdirs.setdefault(posixpath.dirname(d), []).append(d)
    # Deepest first so a parent sees its subdir hashes.
    order = sorted(children, key=lambda x: x.count("/") + (1 if x else 0), reverse=True)
    for d in order:
        entries = list(children[d])
        for sub in subdirs.get(d, ()):
            entries.append((posixpath.basename(sub) + "/", dir_hashes[sub]))
        m = hashlib.sha1()
        for name, h in sorted(entries):
            m.update(name.encode("utf-8", "replace"))
            m.update(b"\0")
            m.update(h.encode("ascii", "replace"))
            m.update(b"\n")
        dir_hashes[d] = m.hexdigest()
    root_hash = dir_hashes.get("", hashlib.sha1(b"").hexdigest())
    unchanged = {d for d, h in dir_hashes.items() if prev_dirs.get(d) == h}
    res = ScanResult(
        hashes=hashes, dir_hashes=dir_hashes, root_hash=root_hash,
        hashed=hashed, reused=reused, unchanged_dirs=unchanged,
        root_unchanged=bool(previous) and previous.get("root") == root_hash,
    )
    return res, stats


def in_unchanged_subtree(rel: str, unchanged_dirs: set[str]) -> bool:
    d = posixpath.dirname(rel)
    return d in unchanged_dirs
