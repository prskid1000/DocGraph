"""Call-resolution confidence cascade.

Tree-sitter gives us call sites as bare names (plus, since v3, the call's
receiver text). Turning `login` at `api.py:12` into an edge to one specific
`Function` node is a guess; this module makes the guess explicit. Each tier
of the cascade stamps the edge with a confidence and the method that
produced it:

    import_map       0.95  the name (or the receiver) was imported by name
                           from a module that resolves to the candidate's file
    same_module      0.90  the candidate is defined in the calling file
    import_suffix    0.85  the candidate lives in a file the caller imports
    unique_global    0.75  exactly one symbol of that name in the repo
    import_distance  0.55  several candidates; one is strictly closest by
                           import-graph distance / receiver suffix / path
    fuzzy            0.30-0.40  no exact name; case/underscore-insensitive match

A call site that stays ambiguous after every tier is NOT turned into a
CALLS edge -- inventing one is how call graphs hallucinate. Its candidates
are returned instead (the indexer stores them as CALLS_CANDIDATE).

Pure Python, no Cypher: the indexer builds the indexes from the DB and the
cache and hands them in.
"""
from __future__ import annotations

import posixpath
from collections import defaultdict, deque
from dataclasses import dataclass, field

CONFIDENCE: dict[str, float] = {
    "scip": 1.0,
    "import_map": 0.95,
    "same_module": 0.90,
    "import_suffix": 0.85,
    "unique_global": 0.75,
    "import_distance": 0.55,
    "common_name": 0.40,
    "fuzzy": 0.40,
    "fuzzy_ranked": 0.30,
}

# Method names so common on builtin containers / std objects that a unique
# repo-wide definition is weak evidence (`d.get(...)` is almost never a call
# to the one `get` the repo defines).
COMMON_METHOD_NAMES = frozenset({
    "get", "set", "put", "pop", "add", "append", "extend", "insert", "remove",
    "update", "items", "keys", "values", "copy", "clear", "sort", "index",
    "count", "join", "split", "strip", "lstrip", "rstrip", "replace", "format",
    "encode", "decode", "lower", "upper", "startswith", "endswith", "find",
    "read", "write", "close", "open", "send", "recv", "run", "start", "stop",
    "wait", "list", "load", "loads", "dump", "dumps", "match", "search", "sub",
    "exists", "mkdir", "resolve", "render", "then", "catch", "push", "map",
    "filter", "reduce", "forEach", "toString", "log", "info", "debug", "error",
    "warning", "execute", "fetch", "call", "apply", "next", "emit", "on",
})

SELF_RECEIVERS = frozenset({"self", "this", "cls", "super", "Self"})

_JS_EXTS = (".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs")
_PY_EXTS = (".py", ".pyi")


@dataclass
class Resolution:
    target: tuple[str, int, str] | None       # (label, id, file) or None
    confidence: float = 0.0
    method: str = "unresolved"
    candidates: list[tuple[str, int, str]] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return self.target is None and bool(self.candidates)


def _norm(name: str) -> str:
    return name.replace("_", "").lower()


def _stem(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    for ext in (*_PY_EXTS, *_JS_EXTS):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    if base in ("__init__", "index", "mod"):
        parts = path.split("/")
        if len(parts) >= 2:
            return parts[-2]
    return base


class ModuleIndex:
    """Maps import strings to indexed file paths.

    Handles dotted Python modules (`a.b` -> `a/b.py` / `a/b/__init__.py`,
    matched as a path suffix so a `src/` layout still resolves), Python
    relative imports (`.x`, `..x`), and JS/TS relative specifiers
    (`./x`, `../x`, with extension and `/index.*` probing). Bare package
    names that match no file (`react`, `os`) resolve to nothing -- they
    become Module nodes, as before.
    """

    def __init__(self, paths: list[str]):
        self.paths = set(paths)
        # path without extension -> [paths]; used for suffix matching
        self._by_noext: dict[str, list[str]] = defaultdict(list)
        self._by_tail: dict[str, list[str]] = defaultdict(list)
        for p in paths:
            noext = p
            for ext in (*_PY_EXTS, *_JS_EXTS):
                if p.endswith(ext):
                    noext = p[: -len(ext)]
                    break
            self._by_noext[noext].append(p)
            if noext.endswith("/__init__") or noext.endswith("/index"):
                self._by_noext[noext.rsplit("/", 1)[0]].append(p)
            tail = noext.rsplit("/", 1)[-1]
            self._by_tail[tail].append(p)
            if tail in ("__init__", "index") and "/" in noext:
                self._by_tail[noext.rsplit("/", 2)[-2]].append(p)
        self._memo: dict[tuple[str, str], list[str]] = {}

    def resolve(self, module: str, src_file: str) -> list[str]:
        key = (module, src_file if module.startswith(".") else "")
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        out = self._resolve(module, src_file)
        self._memo[key] = out
        return out

    def _resolve(self, module: str, src_file: str) -> list[str]:
        module = (module or "").strip().strip("'\"<>`")
        if not module:
            return []
        src_dir = posixpath.dirname(src_file)
        # JS/TS relative specifier
        if module.startswith("./") or module.startswith("../") or module.startswith("/"):
            base = posixpath.normpath(posixpath.join(src_dir, module)).lstrip("/")
            cands = [base, *(base + e for e in _JS_EXTS),
                     *(f"{base}/index{e}" for e in _JS_EXTS), *(base + e for e in _PY_EXTS)]
            return [c for c in cands if c in self.paths][:1]
        # Python relative import: leading dots walk up from the file's dir
        if module.startswith("."):
            dots = len(module) - len(module.lstrip("."))
            rest = module[dots:]
            d = src_dir
            for _ in range(dots - 1):
                d = posixpath.dirname(d)
            base = posixpath.join(d, rest.replace(".", "/")) if rest else d
            base = base.lstrip("/")
            return list(dict.fromkeys(self._by_noext.get(base, [])))[:2]
        # Dotted / slash-separated absolute module
        tp = module.replace(".", "/") if "/" not in module else module
        exact = self._by_noext.get(tp)
        if exact:
            return list(dict.fromkeys(exact))[:2]
        # Suffix match (`pkg.mod` imported from inside `src/pkg/mod.py`)
        tail = tp.rsplit("/", 1)[-1]
        out: list[str] = []
        for p in self._by_tail.get(tail, []):
            noext = p
            for ext in (*_PY_EXTS, *_JS_EXTS):
                if p.endswith(ext):
                    noext = p[: -len(ext)]
                    break
            if noext.endswith("/__init__") or noext.endswith("/index"):
                noext = noext.rsplit("/", 1)[0]
            if noext == tp or noext.endswith("/" + tp):
                out.append(p)
        return list(dict.fromkeys(out))[:2]


class SymbolResolver:
    """The cascade. Build once per index pass.

    name_index: name -> [(label, id, file)]
    file_imports: file -> set(imported files)       (module-level imports)
    import_map: file -> {symbol: set(files)}        (`from m import X` / `import {X} from 'm'`)
    recv_map: file -> {alias: set(files)}           (module aliases usable as receivers)
    """

    def __init__(
        self,
        name_index: dict[str, list[tuple[str, int, str]]],
        file_imports: dict[str, set[str]],
        import_map: dict[str, dict[str, set[str]]] | None = None,
        recv_map: dict[str, dict[str, set[str]]] | None = None,
        aliases: dict[str, dict[str, str]] | None = None,
        method_ids: set[int] | None = None,
        external: dict[str, set[str]] | None = None,
    ):
        self.name_index = name_index
        self.aliases = aliases or {}
        # ids of Functions that are methods (qname Class::name): a bare
        # call cannot reach them, an object call cannot reach anything else.
        self.method_ids = method_ids or set()
        # per file: names bound to imports that resolve to NO repo file
        # (`re`, `os`, `np`, `from pathlib import Path`) -- a call on one of
        # those is an external call, never one of our symbols.
        self.external = external or {}
        self.file_imports = file_imports
        self.import_map = import_map or {}
        self.recv_map = recv_map or {}
        self._norm_index: dict[str, list[tuple[str, int, str]]] | None = None
        self._dist_cache: dict[str, dict[str, int]] = {}
        self.stats: dict[str, int] = defaultdict(int)

    # -- helpers -------------------------------------------------------
    def _norm_idx(self) -> dict[str, list[tuple[str, int, str]]]:
        if self._norm_index is None:
            idx: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
            for name, cands in self.name_index.items():
                if len(name) >= 4:
                    idx[_norm(name)].extend(cands)
            self._norm_index = idx
        return self._norm_index

    def import_distance(self, src: str, max_depth: int = 3) -> dict[str, int]:
        """BFS over the (undirected) file-import graph from `src`."""
        hit = self._dist_cache.get(src)
        if hit is not None:
            return hit
        if not hasattr(self, "_undirected"):
            und: dict[str, set[str]] = defaultdict(set)
            for a, bs in self.file_imports.items():
                for b in bs:
                    und[a].add(b)
                    und[b].add(a)
            self._undirected = und
        dist = {src: 0}
        q = deque([src])
        while q:
            cur = q.popleft()
            d = dist[cur]
            if d >= max_depth:
                continue
            for nxt in self._undirected.get(cur, ()):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    q.append(nxt)
        self._dist_cache[src] = dist
        return dist

    @staticmethod
    def _shared_prefix(a: str, b: str) -> int:
        pa, pb = a.split("/")[:-1], b.split("/")[:-1]
        n = 0
        for x, y in zip(pa, pb):
            if x != y:
                break
            n += 1
        return n

    def _rank(self, cands, src_file: str, recv: str | None):
        """Score candidates for the distance tier. Returns [(score, cand)]
        sorted best first."""
        dist = self.import_distance(src_file)
        recv_tail = (recv or "").rsplit(".", 1)[-1].lower()
        scored = []
        for c in cands:
            s = 0.0
            d = dist.get(c[2])
            if d is not None and d > 0:
                s += 3.0 / d
            if recv_tail and recv_tail not in SELF_RECEIVERS and _stem(c[2]).lower() == recv_tail:
                s += 2.0
            s += 0.25 * self._shared_prefix(src_file, c[2])
            scored.append((s, c))
        scored.sort(key=lambda t: -t[0])
        return scored

    def _done(self, res: Resolution) -> Resolution:
        self.stats[res.method] += 1
        return res

    # -- the cascade ---------------------------------------------------
    def resolve(
        self,
        name: str,
        src_file: str,
        prefer_kind: str | None = None,
        recv: str | None = None,
        attr: bool = False,
    ) -> Resolution:
        if not name:
            return self._done(Resolution(None))
        orig = self.aliases.get(src_file, {}).get(name)
        if orig and orig not in self.name_index and name in self.name_index:
            orig = None
        imported = self.import_map.get(src_file, {}).get(name)
        if orig:
            name = orig
        cands = list(self.name_index.get(name, ()))
        if prefer_kind and cands:
            pref = [c for c in cands if c[0] == prefer_kind]
            if pref:
                cands = pref
        if not cands:
            # Fuzzy only for bare / self calls: a CapWords name is almost
            # always an external class (`Request(...)`, `typer.Exit(...)`),
            # and a call on some other object says nothing about our symbols.
            if (name[:1].isupper() or (recv and recv not in SELF_RECEIVERS)
                    or (attr and not recv)):
                return self._done(Resolution(None))
            return self._done(self._fuzzy(name, src_file, prefer_kind, recv))

        obj_recv = bool(attr and recv not in SELF_RECEIVERS)
        self_recv = bool(attr and recv in SELF_RECEIVERS)

        def module_level(c) -> bool:
            return c[0] == "Function" and c[1] not in self.method_ids

        # 1. import map: explicit symbol import, or an imported module alias
        #    used as the receiver (`auth.login()` after `from src import auth`).
        if obj_recv and recv:
            head = recv.split(".", 1)[0]
            rmap = self.recv_map.get(src_file, {})
            mod_files = rmap.get(recv) or rmap.get(head)
            if mod_files:
                hits = [c for c in cands if c[2] in mod_files]
                if len(hits) == 1:
                    return self._done(Resolution(hits[0], CONFIDENCE["import_map"], "import_map"))
            elif head in self.external.get(src_file, ()):
                return self._done(Resolution(None, 0.0, "external"))
        if obj_recv:
            # `x.m()` on an object: only methods (or nested classes) qualify.
            cands = [c for c in cands if not module_level(c)]
            if not cands:
                return self._done(Resolution(None, 0.0, "external"))
        elif self_recv:
            meths = [c for c in cands if not module_level(c)]
            if meths:
                cands = meths
        elif imported:
            hits = [c for c in cands if c[2] in imported]
            if len(hits) == 1:
                return self._done(Resolution(hits[0], CONFIDENCE["import_map"], "import_map"))

        # 2. same module
        same = [c for c in cands if c[2] == src_file]
        if len(same) == 1:
            return self._done(Resolution(same[0], CONFIDENCE["same_module"], "same_module"))
        if len(same) > 1:
            # Same name twice in one file (overloads / methods of two
            # classes). Still the right file; take the first but say so.
            return self._done(Resolution(same[0], CONFIDENCE["import_distance"], "same_module_multi",
                                         candidates=same))
        if not attr:
            # A bare `f()` cannot call a method defined in another file.
            plain = [c for c in cands if not (c[0] == "Function" and c[1] in self.method_ids)]
            if plain:
                cands = plain
            else:
                return self._done(Resolution(None, 0.0, "unresolved"))

        # 3. import suffix: candidate's file is imported by the caller
        imp_files = self.file_imports.get(src_file, set())
        via_imp = [c for c in cands if c[2] in imp_files]
        if len(via_imp) == 1:
            return self._done(Resolution(via_imp[0], CONFIDENCE["import_suffix"], "import_suffix"))
        if len(via_imp) > 1:
            cands = via_imp  # narrow, then fall to the distance tier

        # 4. unique global name
        if len(cands) == 1:
            if attr and name in COMMON_METHOD_NAMES:
                return self._done(Resolution(cands[0], CONFIDENCE["common_name"], "common_name"))
            return self._done(Resolution(cands[0], CONFIDENCE["unique_global"], "unique_global"))

        # 5. suffix / import distance tie-break
        ranked = self._rank(cands, src_file, recv)
        if ranked and ranked[0][0] > 0 and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
            return self._done(Resolution(ranked[0][1], CONFIDENCE["import_distance"], "import_distance",
                                         candidates=[c for _, c in ranked[1:5]]))
        # Ambiguous: keep the candidates, do not invent an edge.
        return self._done(Resolution(None, 0.0, "ambiguous", candidates=[c for _, c in ranked[:5]]))

    def _same_but_underscores(self, name: str, cand: tuple[str, int, str]) -> bool:
        target = self._names_by_id().get(cand[1], "")
        return bool(target) and target.replace("_", "") == name.replace("_", "") \
            and not (target.startswith("__") and target.endswith("__"))

    def _names_by_id(self) -> dict[int, str]:
        if not hasattr(self, "_id_names"):
            self._id_names = {c[1]: n for n, cs in self.name_index.items() for c in cs}
        return self._id_names

    def _fuzzy(self, name: str, src_file: str, prefer_kind: str | None,
               recv: str | None) -> Resolution:
        if len(name) < 5:
            return Resolution(None)
        # Same letters in the same case; only underscores may differ
        # (`check_password` vs `_check_password`, `getUser` vs `get_user`
        # is NOT matched -- case differs).
        key = _norm(name)
        cands = [c for c in self._norm_idx().get(key, ())
                 if c[0] == (prefer_kind or c[0]) and self._same_but_underscores(name, c)]
        if not cands:
            return Resolution(None)
        if len(cands) == 1:
            return Resolution(cands[0], CONFIDENCE["fuzzy"], "fuzzy")
        ranked = self._rank(cands, src_file, recv)
        if ranked[0][0] > 0 and ranked[0][0] > ranked[1][0]:
            return Resolution(ranked[0][1], CONFIDENCE["fuzzy_ranked"], "fuzzy_ranked",
                              candidates=[c for _, c in ranked[1:5]])
        return Resolution(None, 0.0, "ambiguous_fuzzy", candidates=[c for _, c in ranked[:5]])


def candidate_confidence(n: int) -> float:
    """Confidence stamped on each CALLS_CANDIDATE row of an n-way tie."""
    if n <= 0:
        return 0.0
    return round(0.5 / n, 3)
