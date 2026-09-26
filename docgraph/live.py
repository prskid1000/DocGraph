"""Host-resident incremental index state ("live index").

Everything an incremental pass used to rebuild from scratch on every run --
the whole cache.json, a directory walk with ignore matching, the symbol
table read from Kuzu, per-file import maps, and a scan over every cached
raw edge -- lives here and is updated per changed file:

* ShardedCache     per-file parse entries, kept zlib-compressed in memory
                   and in 1024 binary shard files under `.docgraph/cache/`; only
                   dirty shards are rewritten.
* SymbolTable      name / qname / file -> ids, sorted candidate lists
                   (deterministic: by (file, qname), never by node id), the
                   fuzzy (underscore-insensitive) index, method ids.
* ImportIndex      per-file import records (the resolver's file_imports /
                   import_map / recv_map / aliases / external), the
                   undirected import graph for the distance tier, and a
                   module-tail -> files reverse index (who may need to
                   re-resolve when a file is added or removed).
* RefIndex         target name -> files that reference it (plus the fuzzy
                   key), so an incremental pass finds the callers of the
                   names a change touched without scanning every edge.
* scan()           directory walk that reuses cached ignore / kind
                   decisions and takes file size + mtime from `os.scandir`
                   (free on Windows), or no walk at all when the watcher
                   passes the changed paths.

The CLI builds one per run (from the shards + Kuzu); the host keeps one per
root across runs (`RootSlot.live`), so a watcher-triggered one-file pass
touches only that file. Pure Python + numpy; all Cypher stays in db.py.
"""
from __future__ import annotations

import logging
import os
import threading
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

# ~20 files per shard on a 20k-file repo: a 100-file change rewrites ~16 MB
# of the cache, not most of it
SHARDS = 1024
CACHE_VERSION = 4


# ---------------------------------------------------------------- cache ----

def shard_of(rel: str) -> int:
    return zlib.crc32(rel.encode("utf-8", "replace")) & (SHARDS - 1)


class ShardedCache:
    """rel -> entry, persisted as `.docgraph/cache/<nn>.bin` shards.

    Entries are kept ENCODED (zlib level 1 over orjson bytes) with their
    (size, mtime, hash) alongside, and decoded only when a caller asks for
    one: the whole-repo cache of a 200k-function repo is ~22 MB this way
    (173 MB as plain orjson, ~750 MB as Python dicts), a pass only ever
    decodes the files it touches, and decoding all of it (the one-time
    resolve priming) costs ~0.3 s. The stat check of the scan reads
    `meta()` and never decodes.

    Shard file: b"DGC4" u32 count, then per entry u16 len + rel (utf-8),
    i64 size, i64 mtime_ns, u8 len + hash (ascii), u32 len + the encoded
    bytes as held in memory.
    Mutations mark the shard dirty; save() rewrites only dirty shards."""

    MAGIC = b"DGC4"

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self._raw: dict[str, bytes] = {}
        self._meta: dict[str, tuple[int, int, str]] = {}
        self.dirty: set[int] = set()
        self.loaded = False

    # -- mapping --
    def __len__(self) -> int:
        return len(self._raw)

    def __bool__(self) -> bool:
        return bool(self._raw)

    def __contains__(self, rel) -> bool:
        return rel in self._raw

    def __iter__(self):
        return iter(list(self._raw))

    def keys(self):
        return list(self._raw)

    def get(self, rel, default=None):
        raw = self._raw.get(rel)
        if raw is None:
            return default
        return _dec(raw)

    def __getitem__(self, rel):
        return _dec(self._raw[rel])

    def items(self):
        for rel in list(self._raw):
            raw = self._raw.get(rel)
            if raw is not None:
                yield rel, _dec(raw)

    def meta(self, rel) -> tuple[int, int, str] | None:
        return self._meta.get(rel)

    def __setitem__(self, rel, entry: dict) -> None:
        import orjson
        self.dirty.add(shard_of(rel))
        self._raw[rel] = zlib.compress(orjson.dumps(entry), 1)
        self._meta[rel] = (int(entry.get("size") or 0), int(entry.get("mtime") or 0), str(entry.get("hash") or ""))

    def set_stat(self, rel: str, size: int, mtime: int) -> None:
        e = self.get(rel)
        if e is not None:
            e["size"], e["mtime"] = int(size), int(mtime)
            self[rel] = e

    def __delitem__(self, rel) -> None:
        self.dirty.add(shard_of(rel))
        del self._raw[rel]
        self._meta.pop(rel, None)

    def pop(self, rel, *default):
        if rel in self._raw:
            e = self[rel]
            del self[rel]
            return e
        if default:
            return default[0]
        raise KeyError(rel)

    def clear(self) -> None:
        self.dirty.update(range(SHARDS))
        self._raw.clear()
        self._meta.clear()

    # -- io --
    def load(self) -> "ShardedCache":
        import orjson
        self._raw.clear()
        self._meta.clear()
        self.dirty.clear()
        self.loaded = True
        if not self.dir.exists():
            return self
        try:
            meta = orjson.loads((self.dir / "meta.json").read_bytes())
            if meta.get("version") != CACHE_VERSION:
                return self
        except Exception:
            return self
        n_sh = int(meta.get("shards") or SHARDS)
        paths = [self.dir / _shard_name(i, n_sh) for i in range(n_sh)]

        def read(p: Path):
            try:
                return _decode_shard(p.read_bytes())
            except Exception:
                return []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for part in ex.map(read, paths):
                for rel, size, mtime, h, raw in part:
                    self._raw[rel] = raw
                    self._meta[rel] = (size, mtime, h)
        if n_sh != SHARDS:
            # written with another shard count: re-shard on the next save
            for p in paths:
                try:
                    p.unlink()
                except OSError:
                    pass
            self.dirty.update(range(SHARDS))
        return self

    def save(self) -> int:
        """Write the dirty shards; returns how many were written."""
        import orjson
        if not self.dirty:
            return 0
        self.dir.mkdir(parents=True, exist_ok=True)
        by_shard: dict[int, list[str]] = {i: [] for i in self.dirty}
        for rel in self._raw:
            sh = shard_of(rel)
            if sh in by_shard:
                by_shard[sh].append(rel)
        for sh, rels in by_shard.items():
            p = self.dir / _shard_name(sh, SHARDS)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_bytes(_encode_shard([(r, *self._meta.get(r, (0, 0, "")), self._raw[r]) for r in rels]))
            os.replace(tmp, p)
        (self.dir / "meta.json").write_bytes(orjson.dumps({"version": CACHE_VERSION, "shards": SHARDS}))
        n = len(by_shard)
        self.dirty.clear()
        return n

    def save_all(self) -> int:
        self.dirty.update(range(SHARDS))
        return self.save()

    def wipe(self) -> None:
        import shutil
        self._raw.clear()
        self._meta.clear()
        self.dirty.clear()
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)


def _dec(raw: bytes):
    import orjson
    return orjson.loads(zlib.decompress(raw))


def _shard_name(i: int, n: int) -> str:
    return f"{i:02x}.bin" if n <= 256 else f"{i:03x}.bin"


def _encode_shard(rows) -> bytes:
    import struct
    out = [ShardedCache.MAGIC, struct.pack("<I", len(rows))]
    for rel, size, mtime, h, raw in rows:
        rb = rel.encode("utf-8")
        hb = (h or "").encode("ascii", "replace")[:255]
        out.append(struct.pack("<H", len(rb)))
        out.append(rb)
        out.append(struct.pack("<qqB", int(size), int(mtime), len(hb)))
        out.append(hb)
        out.append(struct.pack("<I", len(raw)))
        out.append(raw)
    return b"".join(out)


def _decode_shard(data: bytes) -> list:
    import struct
    if data[:4] != ShardedCache.MAGIC:
        return []
    (n,) = struct.unpack_from("<I", data, 4)
    o = 8
    out = []
    mv = memoryview(data)
    for _ in range(n):
        (lr,) = struct.unpack_from("<H", data, o)
        o += 2
        rel = bytes(mv[o:o + lr]).decode("utf-8")
        o += lr
        size, mtime, lh = struct.unpack_from("<qqB", data, o)
        o += 17
        h = bytes(mv[o:o + lh]).decode("ascii")
        o += lh
        (lp,) = struct.unpack_from("<I", data, o)
        o += 4
        out.append((rel, size, mtime, h, bytes(mv[o:o + lp])))
        o += lp
    return out


# ---------------------------------------------------------- symbol table ----

class SymbolTable:
    """In-memory symbol table, updated per changed file."""

    def __init__(self) -> None:
        self.name_index: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        self.qname_index: dict[str, tuple[str, int]] = {}
        self.file_index: dict[str, int] = {}
        self.by_file: dict[str, list[tuple[str, int, str, str]]] = {}   # rel -> [(label, id, name, qname)]
        self.id_qname: dict[int, str] = {}
        self.id_names: dict[int, str] = {}
        self.method_ids: set[int] = set()
        self._norm: dict[str, list[tuple[str, int, str]]] | None = None
        self.loaded = False

    def _sort(self, name: str) -> None:
        lst = self.name_index.get(name)
        if lst and len(lst) > 1:
            q = self.id_qname
            lst.sort(key=lambda c: (c[2], q.get(c[1], ""), c[0]))

    def add_file(self, rel: str, file_id: int | None, ents: Iterable[tuple[str, int, str, str]],
                 sort: bool = True) -> None:
        ents = list(ents)
        if file_id is not None:
            self.file_index[rel] = int(file_id)
        self.by_file[rel] = ents
        touched = set()
        for label, i, name, qname in ents:
            i = int(i)
            self.qname_index[qname] = (label, i)
            self.name_index[name].append((label, i, rel))
            self.id_qname[i] = qname
            self.id_names[i] = name
            if label == "Function" and qname.count("::") >= 2:
                self.method_ids.add(i)
            touched.add(name)
            if self._norm is not None and len(name) >= 4:
                self._norm[_norm(name)].append((label, i, rel))
        if sort:
            for n in touched:
                self._sort(n)
            if self._norm is not None:
                for n in touched:
                    if len(n) >= 4:
                        self._sort_norm(_norm(n))

    def _sort_norm(self, key: str) -> None:
        lst = self._norm.get(key) if self._norm is not None else None
        if lst and len(lst) > 1:
            q = self.id_qname
            lst.sort(key=lambda c: (c[2], q.get(c[1], ""), c[0]))

    def finish_bulk(self) -> None:
        for n in list(self.name_index):
            self._sort(n)
        self._norm = None

    def remove_files(self, rels: Iterable[str]) -> set[str]:
        """Drop the files' entities; returns the names they carried."""
        names: set[str] = set()
        for rel in rels:
            ents = self.by_file.pop(rel, None)
            self.file_index.pop(rel, None)
            if not ents:
                continue
            for label, i, name, qname in ents:
                names.add(name)
                cur = self.qname_index.get(qname)
                if cur is not None and cur[1] == i:
                    del self.qname_index[qname]
                self.id_qname.pop(i, None)
                self.id_names.pop(i, None)
                self.method_ids.discard(i)
                lst = self.name_index.get(name)
                if lst is not None:
                    lst[:] = [c for c in lst if c[1] != i]
                    if not lst:
                        del self.name_index[name]
                if self._norm is not None and len(name) >= 4:
                    k = _norm(name)
                    nl = self._norm.get(k)
                    if nl is not None:
                        nl[:] = [c for c in nl if c[1] != i]
                        if not nl:
                            del self._norm[k]
        # a qname shared by two entities (duplicate definitions) falls back
        # to the survivor, like a rebuild would
        return names

    def norm_index(self) -> dict[str, list[tuple[str, int, str]]]:
        if self._norm is None:
            idx: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
            for name, cands in self.name_index.items():
                if len(name) >= 4:
                    idx[_norm(name)].extend(cands)
            q = self.id_qname
            for lst in idx.values():
                if len(lst) > 1:
                    lst.sort(key=lambda c: (c[2], q.get(c[1], ""), c[0]))
            self._norm = idx
        return self._norm

    def ids_of_files(self, rels: Iterable[str]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for rel in rels:
            for label, i, _n, _q in self.by_file.get(rel, ()):
                out[label].append(int(i))
            fid = self.file_index.get(rel)
            if fid is not None:
                out["File"].append(int(fid))
        return out


def _norm(name: str) -> str:
    return name.replace("_", "").lower()


def test_target(name: str) -> str:
    """`test_foo` / `testFoo` -> the name a test function points at (the
    TESTS edge rule)."""
    stripped = name or ""
    for prefix in ("test_", "test"):
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):].lstrip("_")
            break
    return stripped


# ----------------------------------------------------------- imports -------

def _mod_tail(mod: str) -> str:
    m = (mod or "").strip("'\"`<>").rstrip("/")
    if not m:
        return ""
    tail = m.rsplit("/", 1)[-1] if "/" in m else m.rsplit(".", 1)[-1]
    tail = tail.rsplit(".", 1)[0] if "/" in m else tail
    return tail.lstrip(".")


class ImportIndex:
    """Per-file import records + the undirected import graph."""

    def __init__(self) -> None:
        self.file_imports: dict[str, set[str]] = {}
        self.import_map: dict[str, dict[str, set[str]]] = {}
        self.recv_map: dict[str, dict[str, set[str]]] = {}
        self.aliases: dict[str, dict[str, str]] = {}
        self.external: dict[str, set[str]] = {}
        self.tails: dict[str, set[str]] = defaultdict(set)      # module tail -> importing files
        self.file_tails: dict[str, set[str]] = {}
        self.importers: dict[str, set[str]] = defaultdict(set)  # file -> files importing it
        self.undirected = _Undirected(self)

    def remove(self, rel: str) -> None:
        for f in self.file_imports.pop(rel, set()):
            s = self.importers.get(f)
            if s is not None:
                s.discard(rel)
                if not s:
                    del self.importers[f]
        self.import_map.pop(rel, None)
        self.recv_map.pop(rel, None)
        self.aliases.pop(rel, None)
        self.external.pop(rel, None)
        for t in self.file_tails.pop(rel, set()):
            s = self.tails.get(t)
            if s is not None:
                s.discard(rel)
                if not s:
                    del self.tails[t]

    def compute(self, rel: str, edges: list[dict], mod_index) -> None:
        """(Re)build the record of one file from its raw edges -- the same
        rules the indexer always applied, per file."""
        self.remove(rel)
        file_imports: set[str] = set()
        import_map: dict[str, set[str]] = defaultdict(set)
        recv_map: dict[str, set[str]] = defaultdict(set)
        aliases: dict[str, str] = {}
        external: set[str] = set()
        tails: set[str] = set()
        for raw in edges:
            kind = raw.get("kind")
            if kind == "IMPORTS":
                mod = raw.get("target_name") or ""
                t = _mod_tail(mod)
                if t:
                    tails.add(t)
                files = [f for f in mod_index.resolve(mod, rel) if f != rel]
                if not files:
                    m = mod.strip("'\"`")
                    if m and not m.startswith("."):
                        external.add(m.split(".", 1)[0].split("/", 1)[0])
                        external.add(m.rsplit(".", 1)[-1].rsplit("/", 1)[-1])
                    continue
                file_imports.update(files)
                m = mod.strip("'\"`")
                tail = m.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0] if "/" in m else m.rsplit(".", 1)[-1]
                if tail:
                    recv_map[tail].update(files)
                recv_map[m].update(files)
            elif kind == "IMPORTS_SYMBOL":
                ex = raw.get("extra") or {}
                sym = raw.get("target_name") or ""
                mod = ex.get("module") or ""
                alias = ex.get("alias") or ""
                if not sym:
                    continue
                if alias:
                    aliases[alias] = sym
                if not mod:
                    continue
                t = _mod_tail(mod)
                if t:
                    tails.add(t)
                if sym:
                    tails.add(sym)
                files = [f for f in mod_index.resolve(mod, rel) if f != rel]
                if not files and not mod.startswith("."):
                    external.add(alias or sym)
                if files:
                    import_map[sym].update(files)
                    if alias:
                        import_map[alias].update(files)
                    if mod.startswith(".") and "/" in mod:
                        recv_map[alias or sym].update(files)
                sub = mod_index.resolve(f"{mod}.{sym}" if not mod.startswith("./") else f"{mod}/{sym}", rel)
                if sub:
                    recv_map[alias or sym].update(sub)
                    file_imports.update(f for f in sub if f != rel)
        if file_imports:
            self.file_imports[rel] = file_imports
            for f in file_imports:
                self.importers[f].add(rel)
        if import_map:
            self.import_map[rel] = dict(import_map)
        if recv_map:
            self.recv_map[rel] = dict(recv_map)
        if aliases:
            self.aliases[rel] = aliases
        if external:
            self.external[rel] = external
        self.file_tails[rel] = tails
        for t in tails:
            self.tails[t].add(rel)


class _Undirected:
    """Read-only undirected view of the import graph (the resolver's
    distance tier): neighbours = files imported by + files importing."""

    def __init__(self, idx: ImportIndex) -> None:
        self._i = idx

    def get(self, rel: str, default=()):
        a = self._i.file_imports.get(rel)
        b = self._i.importers.get(rel)
        if not a and not b:
            return default
        if not b:
            return a
        if not a:
            return b
        return a | b

    def __getitem__(self, rel: str):
        return self.get(rel, set())


# --------------------------------------------------------------- refs ------

REF_KINDS = ("CALLS", "INSTANTIATES", "INHERITS", "DECORATED_BY", "IMPORTS_SYMBOL", "ROUTE", "TOOL")


class RefIndex:
    """target name (and fuzzy key) -> files whose raw edges name it."""

    def __init__(self) -> None:
        self.by_name: dict[str, set[str]] = defaultdict(set)
        self.by_norm: dict[str, set[str]] = defaultdict(set)
        self.file_names: dict[str, set[str]] = {}

    def remove(self, rel: str) -> None:
        for n in self.file_names.pop(rel, set()):
            s = self.by_name.get(n)
            if s is not None:
                s.discard(rel)
                if not s:
                    del self.by_name[n]
            if len(n) >= 4:
                k = _norm(n)
                s2 = self.by_norm.get(k)
                if s2 is not None:
                    s2.discard(rel)
                    if not s2:
                        del self.by_norm[k]

    def add(self, rel: str, edges: list[dict]) -> None:
        self.remove(rel)
        names: set[str] = set()
        for raw in edges:
            if raw.get("kind") in REF_KINDS:
                t = raw.get("target_name")
                if t:
                    names.add(t)
        self.file_names[rel] = names
        for n in names:
            self.by_name[n].add(rel)
            if len(n) >= 4:
                self.by_norm[_norm(n)].add(rel)

    def files_for(self, names: set[str]) -> set[str]:
        out: set[str] = set()
        for n in names:
            out |= self.by_name.get(n, set())
            if len(n) >= 4:
                out |= self.by_norm.get(_norm(n), set())
        return out


# --------------------------------------------------------------- scan ------

class ScanState:
    """Cached per-path decisions of the walker (ignored? which kind?), valid
    for the host's lifetime (ignore specs are fixed at load_config time)."""

    def __init__(self) -> None:
        self.dir_ok: dict[str, bool] = {}
        self.file_kind: dict[str, str | None] = {}


def scan(cfg, state: ScanState, max_bytes: int) -> dict[str, tuple[Path, int, int]]:
    """{logical_rel: (abs_path, size, mtime_ns)} of every indexable file.
    Uses os.scandir (stat data comes with the listing on Windows) and the
    cached decisions; only names never seen before pay the ignore match and
    the binary sniff."""
    out: dict[str, tuple[Path, int, int]] = {}
    for root, prefix in cfg.roots_with_prefix():
        _walk(cfg, state, Path(root), prefix, str(root), "", max_bytes, out)
    return out


def _walk(cfg, state: ScanState, root: Path, prefix: str, start: str, start_rel: str,
          max_bytes: int, out: dict) -> None:
    from docgraph.parse import classify_file
    text_fallback = bool(getattr(cfg, "text_fallback", True))
    stack = [(start, start_rel)]
    while stack:
        dpath, drel = stack.pop()
        try:
            it = os.scandir(dpath)
        except OSError:
            continue
        with it:
            for e in it:
                name = e.name
                rel = f"{drel}/{name}" if drel else name
                try:
                    is_dir = e.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                key = f"{prefix}{rel}"
                if is_dir:
                    ok = state.dir_ok.get(key)
                    if ok is None:
                        ok = not cfg.is_ignored(f"{rel}/", root=root)
                        state.dir_ok[key] = ok
                    if ok:
                        stack.append((e.path, rel))
                    continue
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if st.st_size > max_bytes:
                    continue
                if key in state.file_kind:
                    kind = state.file_kind[key]
                else:
                    kind = None
                    if not cfg.is_ignored(rel, root=root):
                        kind = classify_file(Path(e.path), sniff_known=False)
                    state.file_kind[key] = kind
                if kind is None or (kind.startswith("text:") and not text_fallback):
                    continue
                out[key] = (Path(e.path), int(st.st_size), int(st.st_mtime_ns))


def classify_paths(cfg, state: ScanState, paths: Iterable[Path], max_bytes: int
                   ) -> tuple[dict[str, tuple[Path, int, int]], set[str]]:
    """Watcher fast path: ({rel: (path, size, mtime)} of the indexable
    event paths that exist, {rels that no longer exist}). A deleted
    directory is reported as its own rel; the caller expands it."""
    from docgraph.parse import classify_file
    present: dict[str, tuple[Path, int, int]] = {}
    gone: set[str] = set()
    roots = [(Path(r).resolve(), p) for r, p in cfg.roots_with_prefix()]
    for raw in paths:
        p = Path(raw)
        try:
            ap = p.resolve()
        except OSError:
            ap = p
        owner = None
        for r, prefix in roots:
            try:
                rel = ap.relative_to(r).as_posix()
                owner = (r, prefix, rel)
                break
            except ValueError:
                continue
        if owner is None:
            continue
        r, prefix, rel = owner
        key = f"{prefix}{rel}"
        try:
            st = ap.stat()
        except OSError:
            gone.add(key)
            continue
        if ap.is_dir():
            # a directory that appeared: walk it
            if not cfg.is_ignored(f"{rel}/", root=r):
                _walk(cfg, state, r, prefix, str(ap), rel, max_bytes, present)
            continue
        if st.st_size > max_bytes or cfg.is_ignored(rel, root=r):
            continue
        kind = classify_file(ap, sniff_known=False)
        state.file_kind[key] = kind
        if kind is None or (kind.startswith("text:") and not getattr(cfg, "text_fallback", True)):
            continue
        present[key] = (ap, int(st.st_size), int(st.st_mtime_ns))
    return present, gone


# --------------------------------------------------------------- live ------

class LiveIndex:
    """All incremental state of one root. `lock` serializes users."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.cache = ShardedCache(Path(cfg.data_dir) / "cache")
        self.symtab = SymbolTable()
        self.imports = ImportIndex()
        self.refs = RefIndex()
        self.scan_state = ScanState()
        self.mod_index = None
        self.mod_paths: frozenset[str] = frozenset()
        self.modules: dict[str, dict] = {}          # Module name -> row
        self.modules_loaded = False
        self.resolve_ready = False                  # imports + refs built for every file
        self.tiles = None                           # (manifest, arrays) of the sidecar, in memory
        self.schema_version: int | None = None
        self.scan_ready = False                     # scan_state primed by a full walk
        self.last_walk = 0.0                        # time of that walk (watcher paths trusted ~10 min)
        self.next_id = 0                            # id allocator position after the last pass
        self.vector_ok = False                      # HNSW indexes present and current
        self.vector_status: dict = {}
        self.kw = None                              # kwindex.KeywordIndex
        self.tests: dict[str, set[int]] | None = None   # stripped test name -> test function ids
        self.test_names: dict[int, str] = {}
        self.generation = 0                         # bumped by every index pass
        self.git_roots = None                       # roots that are git work trees (cached)
        # Host only: a parse process pool kept between passes (spawning one
        # on Windows costs ~0.5-0.8 s, more than parsing 50 files), shut
        # down after POOL_IDLE_SEC without a pass.
        self.keep_pool = False
        self._pool = None
        self._pool_workers = 0
        self._pool_used = 0.0
        self._pool_timer = None

    POOL_IDLE_SEC = 300.0

    def parse_pool(self, workers: int):
        """The kept process pool (created on first use, recreated when a
        bigger one is asked for or the old one broke)."""
        import time as _time
        from concurrent.futures import ProcessPoolExecutor
        with self.lock:
            if self._pool is not None and (self._pool_workers < workers
                                           or getattr(self._pool, "_broken", False)):
                self.close_pool()
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=workers)
                self._pool_workers = workers
            self._pool_used = _time.time()
            if self._pool_timer is None:
                t = threading.Timer(self.POOL_IDLE_SEC, self._pool_idle_check)
                t.daemon = True
                self._pool_timer = t
                t.start()
            return self._pool

    def _pool_idle_check(self) -> None:
        import time as _time
        with self.lock:
            self._pool_timer = None
            if self._pool is None:
                return
            idle = _time.time() - self._pool_used
            if idle >= self.POOL_IDLE_SEC:
                self.close_pool()
                return
            t = threading.Timer(self.POOL_IDLE_SEC - idle + 1.0, self._pool_idle_check)
            t.daemon = True
            self._pool_timer = t
            t.start()

    def close_pool(self) -> None:
        pool, self._pool = self._pool, None
        self._pool_workers = 0
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def reset(self) -> None:
        self.cache = ShardedCache(Path(self.cfg.data_dir) / "cache")
        self.symtab = SymbolTable()
        self.imports = ImportIndex()
        self.refs = RefIndex()
        self.mod_index = None
        self.mod_paths = frozenset()
        self.modules = {}
        self.modules_loaded = False
        self.resolve_ready = False
        self.tiles = None
        self.scan_ready = False
        self.last_walk = 0.0
        self.next_id = 0
        self.vector_ok = False
        self.vector_status = {}
        self.kw = None
        self.tests = None
        self.test_names = {}

    def ensure_kw(self, load: bool = True):
        from docgraph.kwindex import KeywordIndex
        if self.kw is None:
            self.kw = KeywordIndex(Path(self.cfg.data_dir))
            if load:
                self.kw.load()
        return self.kw

    def ensure_tests(self, db) -> dict[str, set[int]]:
        if self.tests is None:
            self.tests = defaultdict(set)
            self.test_names = {}
            for r in db.test_function_rows():
                self.add_test(int(r["id"]), r.get("name") or "")
        return self.tests

    def add_test(self, i: int, name: str) -> None:
        k = test_target(name)
        if k and self.tests is not None:
            self.tests[k].add(i)
            self.test_names[i] = k

    def remove_tests(self, ids) -> None:
        if self.tests is None:
            return
        for i in ids:
            k = self.test_names.pop(int(i), None)
            if k is not None:
                s = self.tests.get(k)
                if s is not None:
                    s.discard(int(i))
                    if not s:
                        del self.tests[k]

    def ensure_cache(self) -> ShardedCache:
        if not self.cache.loaded:
            self.cache.load()
        return self.cache

    def ensure_symtab(self, db) -> SymbolTable:
        if self.symtab.loaded:
            return self.symtab
        st = SymbolTable()
        rows = db.symbol_rows()          # [(label, id, name, qname, file)] sorted by id
        by_file: dict[str, list] = defaultdict(list)
        for label, i, name, qname, f in rows:
            by_file[f].append((label, i, name, qname))
        import sys as _sys
        for path, fid in db.file_ids().items():
            st.file_index[_sys.intern(path)] = fid
        for f, ents in by_file.items():
            st.add_file(f, st.file_index.get(f), ents, sort=False)
        for f in st.file_index:
            st.by_file.setdefault(f, [])
        st.finish_bulk()
        st.loaded = True
        self.symtab = st
        return st

    def ensure_modules(self, db) -> dict[str, dict]:
        if not self.modules_loaded:
            self.modules = {r["name"]: {"id": r["id"], "name": r["name"], "language": ""}
                            for r in db.module_rows()}
            self.modules_loaded = True
        return self.modules

    def ensure_mod_index(self):
        from docgraph.resolve import ModuleIndex
        paths = frozenset(self.symtab.file_index)
        if self.mod_index is None or paths != self.mod_paths:
            self.mod_index = ModuleIndex(sorted(paths))
            self.mod_paths = paths
        return self.mod_index

    def ensure_resolve(self) -> None:
        """Import records + reverse refs for every cached file."""
        if self.resolve_ready:
            return
        mi = self.ensure_mod_index()
        self.imports = ImportIndex()
        self.refs = RefIndex()
        for rel, entry in self.cache.items():
            edges = entry.get("edges") or []
            self.imports.compute(rel, edges, mi)
            self.refs.add(rel, edges)
        self.resolve_ready = True
