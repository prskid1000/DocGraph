"""Kuzu storage layer. One file per repo at .docgraph/graph.kuzu.

Schema covers all Tier 1-4 relationships.
"""
from __future__ import annotations

import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

import kuzu
import pyarrow as pa

# Node tables — all entities share an integer pk for fast joins; qname is the
# stable human-readable identifier.
#
# Embedding columns are `FLOAT[{dim}]` (v4; v3 used DOUBLE) and substituted at init time from
# `GraphDB.embedding_dim` so the DB matches the chosen embedding model
# (BGE small = 384, mpnet = 768, e5-large = 1024 …). A schema mismatch is
# unrecoverable — Kuzu won't auto-resize a fixed-length array column — so
# switching model requires `docgraph admin clear` + full reindex.
NODE_DDL = [
    """CREATE NODE TABLE IF NOT EXISTS File(
        id INT64,
        path STRING,
        language STRING,
        lines INT64,
        hash STRING,
        pagerank DOUBLE,
        x DOUBLE,
        y DOUBLE,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Module(
        id INT64,
        name STRING,
        language STRING,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Class(
        id INT64,
        name STRING,
        qname STRING,
        file STRING,
        line_start INT64,
        line_end INT64,
        body STRING,
        kind STRING,
        llm_doc STRING,
        pagerank DOUBLE,
        ehash STRING,
        terms STRING,
        x DOUBLE,
        y DOUBLE,
        first_seen_commit STRING,
        first_seen_ts INT64,
        last_changed_commit STRING,
        last_changed_ts INT64,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Function(
        id INT64,
        name STRING,
        qname STRING,
        file STRING,
        line_start INT64,
        line_end INT64,
        body STRING,
        signature STRING,
        is_method BOOLEAN,
        is_test BOOLEAN,
        llm_doc STRING,
        pagerank DOUBLE,
        ehash STRING,
        terms STRING,
        x DOUBLE,
        y DOUBLE,
        first_seen_commit STRING,
        first_seen_ts INT64,
        last_changed_commit STRING,
        last_changed_ts INT64,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Variable(
        id INT64,
        name STRING,
        qname STRING,
        file STRING,
        line INT64,
        scope STRING,
        x DOUBLE,
        y DOUBLE,
        PRIMARY KEY (id)
    )""",
    # Sub-function chunks. Long entity bodies get split into sub-chunks so
    # semantic search has finer recall than one-vector-per-1000-line-class.
    # parent_qname / parent_label / file kept on each chunk so incremental
    # delete-by-file works the same way as for entities.
    """CREATE NODE TABLE IF NOT EXISTS Chunk(
        id INT64,
        parent_qname STRING,
        parent_label STRING,
        file STRING,
        idx INT64,
        body STRING,
        ehash STRING,
        line_start INT64,
        line_end INT64,
        terms STRING,
        PRIMARY KEY (id)
    )""",
    # Embeddings (schema v5) live in side tables keyed by the entity's id,
    # carrying the HNSW index: an entity row is written without paying the
    # vector-index insertion (~5 ms per row on a 200k-row index), and the host
    # adds the vectors in the background (Indexer.vector_sink).
    """CREATE NODE TABLE IF NOT EXISTS FnVec(id INT64, embedding FLOAT[{dim}], PRIMARY KEY (id))""",
    """CREATE NODE TABLE IF NOT EXISTS ClsVec(id INT64, embedding FLOAT[{dim}], PRIMARY KEY (id))""",
    """CREATE NODE TABLE IF NOT EXISTS ChunkVec(id INT64, embedding FLOAT[{dim}], PRIMARY KEY (id))""",
    # Server-side communities (Louvain over the resolved call/type graph).
    # Wiped + recomputed on every index pass that dirties the graph, like
    # PageRank. `top_members` / `files` are JSON-encoded lists.
    """CREATE NODE TABLE IF NOT EXISTS Community(
        id INT64,
        name STRING,
        size INT64,
        cohesion DOUBLE,
        top_members STRING,
        files STRING,
        pagerank DOUBLE,
        x DOUBLE,
        y DOUBLE,
        r DOUBLE,
        PRIMARY KEY (id)
    )""",
    # Framework HTTP routes (aiohttp / FastAPI / Flask / Express) and MCP
    # tools / resources / prompts. Keyed by `file` like every other entity
    # so the per-file delta can DETACH DELETE them.
    """CREATE NODE TABLE IF NOT EXISTS Route(
        id INT64,
        name STRING,
        method STRING,
        path STRING,
        framework STRING,
        file STRING,
        line INT64,
        handler STRING,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Tool(
        id INT64,
        name STRING,
        kind STRING,
        framework STRING,
        file STRING,
        line INT64,
        handler STRING,
        PRIMARY KEY (id)
    )""",
]

# Edge tables — Kuzu requires explicit FROM/TO node tables. We declare the
# realistic combinations only.
EDGE_DDL = [
    # Tier 1 — Structural
    "CREATE REL TABLE IF NOT EXISTS CONTAINS(FROM File TO Class, FROM File TO Function, FROM File TO Variable, FROM Class TO Function, FROM Class TO Variable, FROM Class TO Class)",
    "CREATE REL TABLE IF NOT EXISTS IMPORTS(FROM File TO File, FROM File TO Module)",
    "CREATE REL TABLE IF NOT EXISTS IMPORTS_SYMBOL(FROM File TO Class, FROM File TO Function)",
    # Tier 2 — Behavioral
    # Resolved edges carry `confidence` (0..1) + `method` (which tier of the
    # resolution cascade produced them -- see resolve.py). Ambiguous call
    # sites are NOT turned into CALLS; their candidates go to CALLS_CANDIDATE.
    "CREATE REL TABLE IF NOT EXISTS CALLS(FROM Function TO Function, line INT64, confidence DOUBLE, method STRING)",
    "CREATE REL TABLE IF NOT EXISTS CALLS_CANDIDATE(FROM Function TO Function, line INT64, confidence DOUBLE, method STRING)",
    "CREATE REL TABLE IF NOT EXISTS INSTANTIATES(FROM Function TO Class, line INT64, confidence DOUBLE, method STRING)",
    "CREATE REL TABLE IF NOT EXISTS REFERENCES_(FROM Function TO Class, FROM Function TO Variable, FROM Function TO Function, line INT64)",
    "CREATE REL TABLE IF NOT EXISTS RETURNS(FROM Function TO Class)",
    # Tier 3 — Type system
    "CREATE REL TABLE IF NOT EXISTS INHERITS(FROM Class TO Class, confidence DOUBLE, method STRING)",
    "CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(FROM Class TO Class)",
    "CREATE REL TABLE IF NOT EXISTS OVERRIDES(FROM Function TO Function)",
    "CREATE REL TABLE IF NOT EXISTS DECORATED_BY(FROM Function TO Function, FROM Class TO Function)",
    # Tier 4 — Differentiators
    "CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(FROM Function TO Function, FROM Class TO Class, score DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS CO_CHANGED_WITH(FROM File TO File, count INT64)",
    "CREATE REL TABLE IF NOT EXISTS TESTS(FROM Function TO Function, FROM Function TO Class)",
    "CREATE REL TABLE IF NOT EXISTS CONTAINS_CHUNK(FROM Function TO Chunk, FROM Class TO Chunk, FROM File TO Chunk)",
    # External-link structure: BFS parent→child hyperlinks from the web crawler.
    "CREATE REL TABLE IF NOT EXISTS LINKS_TO(FROM File TO File)",
    # Communities + framework maps
    "CREATE REL TABLE IF NOT EXISTS MEMBER_OF(FROM Function TO Community, FROM Class TO Community, FROM File TO Community)",
    "CREATE REL TABLE IF NOT EXISTS HANDLES(FROM Route TO Function, FROM Tool TO Function)",
]

# Bumped whenever the on-disk schema or the cache.json entry shape changes.
# `Indexer.index_all` compares it with state.json["schema_version"] and
# forces a full reindex on mismatch (an incremental run over an old-shape
# DB would leave pre-upgrade edges without confidence, nodes without
# ehash, etc.).
SCHEMA_VERSION = 5

# Columns added after the first public schema. `migrate()` ALTERs them onto
# an existing DB opened read-write so an old DB never hard-errors on a new
# query; readers additionally probe `table_props()` before using them.
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "CALLS": [("confidence", "DOUBLE"), ("method", "STRING")],
    "INSTANTIATES": [("confidence", "DOUBLE"), ("method", "STRING")],
    "INHERITS": [("confidence", "DOUBLE"), ("method", "STRING")],
    "Function": [("ehash", "STRING"), ("first_seen_commit", "STRING"),
                 ("first_seen_ts", "INT64"), ("last_changed_commit", "STRING"),
                 ("last_changed_ts", "INT64")],
    "Class": [("ehash", "STRING"), ("first_seen_commit", "STRING"),
              ("first_seen_ts", "INT64"), ("last_changed_commit", "STRING"),
              ("last_changed_ts", "INT64")],
    "Chunk": [("ehash", "STRING"), ("line_start", "INT64"), ("line_end", "INT64"),
              ("terms", "STRING")],
}
# v4 layout / keyword columns; ALTERed onto an old DB like the ones above.
for _t in ("Function", "Class"):
    _ADDED_COLUMNS[_t] += [("terms", "STRING"), ("x", "DOUBLE"), ("y", "DOUBLE")]
_ADDED_COLUMNS["File"] = [("x", "DOUBLE"), ("y", "DOUBLE")]
_ADDED_COLUMNS["Variable"] = [("x", "DOUBLE"), ("y", "DOUBLE")]
_ADDED_COLUMNS["Community"] = [("x", "DOUBLE"), ("y", "DOUBLE"), ("r", "DOUBLE")]


class DatabaseBusy(RuntimeError):
    """Raised when a query hits a connection that's been closed because a
    writer (watcher reindex / index / wiki) currently owns the
    file's exclusive Kuzu lock. Routes catch this and return 503 +
    Retry-After so the client can poll until the writer releases.

    Kuzu enforces a per-DB-file write lock — we cannot keep an RO handle
    open while a writer is active in another connection — so the only
    sane behavior during a writer-held window is to refuse reads with a
    well-typed error, not crash with AttributeError on a None conn."""


# Kuzu reserves `max_db_size` bytes of *virtual* address space per open
# Database (default 8 TB). Windows gives a process 128 TB, so ~16 open
# databases -- a host with many roots, or a test session -- fail with
# "VirtualAlloc ... failed". 1 TB is still orders of magnitude above any
# code graph.
MAX_DB_SIZE = 1 << 40


def _default_buffer_pool() -> int:
    """Kuzu's own default buffer pool is 80% of physical RAM, which a big
    index happily fills. Bound it to a quarter of RAM, at most 4 GB (Kuzu
    spills to disk beyond the pool)."""
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            total = int(m.ullTotalPhys)
        else:
            raise OSError
    except Exception:
        try:
            import os as _os
            total = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
        except Exception:
            return 0
    return int(min(4 << 30, max(512 << 20, total // 4)))


BUFFER_POOL_SIZE = _default_buffer_pool()

# `--db-buffer-mb`: the Kuzu buffer pool per open database. 0 = scaled to
# the database: half its size on disk, clamped to [256, 512] MB (a 720 MB
# graph of 200k functions gets 360 MB). Kuzu keeps every page it has read
# in the pool until the database closes, so the pool is the ceiling of its
# resident memory; queries past it evict pages, they do not fail.
BUFFER_MIN_MB = 256
BUFFER_MAX_MB = 512
_BUFFER_MB = 0


def set_buffer_pool_mb(mb: int | float | None) -> None:
    """Process-wide buffer pool size for databases opened from now on
    (0 / None = scaled to the database)."""
    global _BUFFER_MB
    _BUFFER_MB = max(0, int(mb or 0))


def _dir_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
        return total
    except OSError:
        return 0


def buffer_pool_bytes(db_path: Path, mb: int | None = None) -> int:
    mb = _BUFFER_MB if mb is None else int(mb)
    if mb > 0:
        return int(mb) << 20
    size = _dir_size(Path(db_path))
    auto = min(BUFFER_MAX_MB << 20, max(BUFFER_MIN_MB << 20, size // 2))
    cap = BUFFER_POOL_SIZE or auto
    return int(min(auto, cap))


_PK_IN = re.compile(r"\b([A-Za-z_]\w*)\.id IN \$ids\b")
_PK_EQ = re.compile(r"\b([A-Za-z_]\w*)\.id = \$id\b")


class EdgeJournal:
    """What an incremental pass changed in the graph, for patching derived
    data (the tile sidecar) without re-reading the whole graph:
    ("add", rel, from_ids, to_ids, conf), ("del_pairs", rel, a, b),
    ("del_out", rel, src_ids), ("del_kind", rel), ("del_nodes", ids)."""

    def __init__(self) -> None:
        self.ops: list[tuple] = []

    def __len__(self) -> int:
        return len(self.ops)


class GraphDB:
    def __init__(self, db_path: Path, embedding_dim: int = 384, read_only: bool = False,
                 buffer_pool_mb: int | None = None, bulk: bool = False):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        # `bulk`: a full / very large pass. Kuzu 0.11 reports a spurious
        # "duplicated primary key" while bulk-inserting 200k rows into a
        # vector table under a 256 MB pool (reproducible; fine at 4 GB), so
        # bulk loads get the large pool; the host drops the handle after the
        # pass and reads through a capped one again.
        self.bulk = bool(bulk)
        self.buffer_pool_bytes = buffer_pool_bytes(self.db_path, buffer_pool_mb)
        if self.bulk:
            self.buffer_pool_bytes = max(self.buffer_pool_bytes, BUFFER_POOL_SIZE)
        self.db = kuzu.Database(str(self.db_path), read_only=read_only, max_db_size=MAX_DB_SIZE,
                                buffer_pool_size=self.buffer_pool_bytes)
        self.conn = kuzu.Connection(self.db)
        # Per-node-table id sets, populated lazily on first edge insert. Used
        # to filter dangling-endpoint rows before COPY FROM (which errors hard
        # on unknown PKs, vs. the old MATCH+CREATE which silently dropped).
        # `insert_nodes` extends the cache so freshly inserted nodes are seen.
        self._known_ids: dict[str, set[int]] = {}
        self._props_cache: dict[str, set[str]] = {}
        self._col_types: dict[str, dict[str, str]] = {}
        self.read_only = read_only
        self.journal: EdgeJournal | None = None
        # vector_sink(label, [{id, embedding}]) takes the vectors of inserted
        # entities instead of writing them now (the host defers them)
        self.vector_sink = None
        # Per-thread side connections (see thread_conn) for concurrent reads.
        import threading as _th
        self._tls = _th.local()
        self._side: list = []
        self._side_lock = _th.Lock()

    def init_schema(self) -> None:
        # NODE_DDL templates contain `{dim}` placeholders for embedding columns
        # so the on-disk schema matches whatever model the user picked.
        for ddl in NODE_DDL:
            self.conn.execute(ddl.format(dim=self.embedding_dim))
        for ddl in EDGE_DDL:
            self.conn.execute(ddl)
        self.migrate()

    def migrate(self) -> None:
        """ALTER columns added in later schema versions onto an existing
        table. Idempotent; no-op on a fresh DB. Values of the new columns
        stay NULL until the next full reindex writes them."""
        for table, cols in _ADDED_COLUMNS.items():
            have = self.table_props(table, refresh=True)
            if not have:
                continue
            for col, typ in cols:
                if col in have:
                    continue
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD {col} {typ}")
                except Exception:
                    pass
            self._props_cache.pop(table, None)

    def table_props(self, table: str, refresh: bool = False) -> set[str]:
        """Property names of a node/rel table (empty if the table is
        missing). Cached per connection; readers use it to degrade
        gracefully on a DB written by an older schema version."""
        if not refresh and table in self._props_cache:
            return self._props_cache[table]
        props: set[str] = set()
        try:
            for r in self.fetch_all(f"CALL table_info('{table}') RETURN *"):
                name = r.get("name")
                if name:
                    props.add(str(name))
        except Exception:
            props = set()
        self._props_cache[table] = props
        return props

    def has_table(self, table: str) -> bool:
        if self.table_props(table):
            return True
        # Rel tables without properties report no columns in table_info.
        return table in self.table_names()

    def table_names(self) -> set[str]:
        names = self._props_cache.get("#tables")
        if names is None:
            try:
                names = {str(r.get("name")) for r in self.fetch_all("CALL show_tables() RETURN *")}
            except Exception:
                names = set()
            self._props_cache["#tables"] = names
        return names

    # ---- index-time helpers for the newer tables -------------------------

    def replace_communities(self, communities: list[dict],
                            members: dict[str, list[dict]]) -> None:
        """Wipe Community + MEMBER_OF and write a fresh partition.
        communities: rows for the Community table. members: {from_label:
        [{from_id, to_id}]} for MEMBER_OF."""
        try:
            self.execute("MATCH ()-[r:MEMBER_OF]->() DELETE r")
            self.execute("MATCH (c:Community) DETACH DELETE c")
        except Exception:
            pass
        self._known_ids.pop("Community", None)
        if communities:
            self.insert_nodes("Community", communities)
        for from_label, rows in members.items():
            if rows:
                self.insert_edges("MEMBER_OF", from_label, "Community", rows)

    def upgrade_calls(self, rows: list[dict]) -> int:
        """Raise existing CALLS edges to a precise confidence/method (SCIP).
        rows: [{a, b, confidence, method}]. Returns the number of rows sent."""
        import importlib.util  # noqa: F401
        if not rows:
            return 0
        up_arrow = pa.table({  # noqa: F841 - referenced by name in the query
            "_a": pa.array([int(r["a"]) for r in rows], pa.int64()),
            "_b": pa.array([int(r["b"]) for r in rows], pa.int64()),
            "_c": pa.array([float(r["confidence"]) for r in rows], pa.float64()),
            "_m": pa.array([str(r["method"]) for r in rows], pa.string()),
        })
        self.execute(
            "LOAD FROM up_arrow WITH _a AS _a, _b AS _b, _c AS _c, _m AS _m "
            "MATCH (a:Function)-[r:CALLS]->(b:Function) WHERE a.id = _a AND b.id = _b "
            "SET r.confidence = _c, r.method = _m")
        return len(rows)

    def set_history(self, label: str, rows: list[dict]) -> None:
        """Update history columns in place: rows [{id, fc, fts, lc, lts}]."""
        import numpy as np
        if not rows:
            return
        self.set_node_values(label, np.array([r["id"] for r in rows], np.int64), {
            "first_seen_commit": np.array([r["fc"] for r in rows], dtype=object),
            "first_seen_ts": np.array([int(r["fts"]) for r in rows], np.int64),
            "last_changed_commit": np.array([r["lc"] for r in rows], dtype=object),
            "last_changed_ts": np.array([int(r["lts"]) for r in rows], np.int64),
        })

    def community_graph(self) -> tuple[dict[int, dict], list[tuple[int, int, float]]]:
        """Nodes + weighted edges for community detection (communities.py)."""
        nodes: dict[int, dict] = {}
        for label in ("Function", "Class"):
            for r in self.fetch_all(
                f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, n.file AS file, "
                f"coalesce(n.pagerank, 0.0) AS pr"
            ):
                nodes[r["id"]] = {"label": label, "name": r["name"], "file": r["file"],
                                  "pagerank": float(r["pr"] or 0.0)}
        for r in self.fetch_all(
            "MATCH (n:File) RETURN n.id AS id, n.path AS path, coalesce(n.pagerank, 0.0) AS pr"
        ):
            nodes[r["id"]] = {"label": "File", "name": r["path"], "file": r["path"],
                              "pagerank": float(r["pr"] or 0.0)}
        edges: list[tuple[int, int, float]] = []
        conf_rel = {t: ("confidence" in self.table_props(t)) for t in ("CALLS", "INSTANTIATES", "INHERITS")}
        for rel in ("CALLS", "INSTANTIATES", "INHERITS"):
            w = "coalesce(r.confidence, 1.0)" if conf_rel[rel] else "1.0"
            for r in self.fetch_all(f"MATCH (a)-[r:{rel}]->(b) RETURN a.id AS a, b.id AS b, {w} AS w"):
                edges.append((r["a"], r["b"], float(r["w"] or 1.0)))
        for r in self.fetch_all("MATCH (a)-[r:CONTAINS]->(b) RETURN a.id AS a, b.id AS b"):
            edges.append((r["a"], r["b"], 0.5))
        for r in self.fetch_all("MATCH (a:File)-[r:IMPORTS]->(b:File) RETURN a.id AS a, b.id AS b"):
            edges.append((r["a"], r["b"], 0.3))
        return nodes, edges

    def entity_spans(self, files: list[str]) -> list[dict]:
        """[{label, id, file, s, e}] for Function/Class nodes in `files`."""
        out: list[dict] = []
        if not files:
            return out
        for label in ("Function", "Class"):
            for r in self.fetch_all(
                f"MATCH (n:{label}) WHERE n.file IN $files "
                f"RETURN n.id AS id, n.file AS file, n.line_start AS s, n.line_end AS e",
                {"files": files},
            ):
                r["label"] = label
                out.append(r)
        return out

    def embeddings_by_ehash(self, label: str, hashes: list[str],
                            files: list[str] | None = None) -> dict[str, list]:
        """Existing vectors for content hashes (the embedding cache). With
        `files`, only nodes in those files are searched (the pre-delete
        harvest); otherwise the whole table."""
        if not hashes or "ehash" not in self.table_props(label):
            return {}
        if files is not None:
            q = (f"MATCH (n:{label}) WHERE n.file IN $files AND n.ehash IS NOT NULL "
                 f"RETURN n.id AS id, n.ehash AS h")
            params: dict = {"files": files}
        else:
            q = f"MATCH (n:{label}) WHERE n.ehash IN $hs RETURN n.id AS id, n.ehash AS h"
            params = {"hs": hashes}
        want = set(hashes)
        seen: set[str] = set()
        pairs = []
        for r in self.fetch_all(q, params):
            h = r.get("h")
            if h in want and h not in seen:
                seen.add(h)
                pairs.append((int(r["id"]), h))
        return self._vectors_by_hash_rows(label, pairs)

    # ---- search indexes (Kuzu's statically linked vector + FTS extensions) ----
    #
    # Kuzu 0.11 links VECTOR and FTS into the core, so nothing is INSTALLed
    # (no network, offline-safe). Both index kinds are maintained by Kuzu on
    # every CREATE / DELETE, so an incremental pass needs no rebuild; a full
    # pass creates them once after the bulk load (much faster than inserting
    # into a live HNSW graph). Embedding columns cannot be SET while a vector
    # index exists -- rows are always delete + insert, which the indexer
    # already does.

    VECTOR_INDEXES = {"Function": "fn_vec", "Class": "cls_vec", "Chunk": "chunk_vec"}
    VEC_TABLES = {"Function": "FnVec", "Class": "ClsVec", "Chunk": "ChunkVec"}

    def vec_table(self, label: str) -> str:
        """Table holding `label`'s embeddings: the side table (schema v5),
        else the entity table itself (an older database)."""
        t = self.VEC_TABLES.get(label)
        if t and self.has_table(t):
            return t
        return label
    # Kuzu FTS indexes of schema v4. Not created any more (deleting a row
    # inserted after an FTS index was built crashes Kuzu 0.11; keywords are
    # served by docgraph.kwindex) -- only dropped when found.
    FTS_INDEXES = {"Function": ("fn_fts", ["name", "terms", "body"]),
                   "Class": ("cls_fts", ["name", "terms", "body"]),
                   "Chunk": ("chunk_fts", ["terms", "body"])}

    def list_indexes(self) -> dict[str, dict]:
        """{index_name: {table, type, props}} from `show_indexes()`."""
        out: dict[str, dict] = {}
        try:
            for r in self.fetch_all("CALL show_indexes() RETURN *"):
                name = r.get("index name") or r.get("index_name") or ""
                if name:
                    out[str(name)] = {"table": r.get("table name") or r.get("table_name"),
                                      "type": r.get("index type") or r.get("index_type"),
                                      "props": r.get("property names") or r.get("property_names")}
        except Exception:
            return {}
        return out

    def ensure_search_indexes(self, on_progress: "Callable[[str], None] | None" = None) -> dict:
        """Create any missing vector index (and drop legacy FTS indexes).
        Returns {vector: {label: ok|error}, fts: {label: "kwindex"}}; never
        raises."""
        have = self.list_indexes()
        status: dict[str, dict[str, str]] = {"vector": {}, "fts": {}}
        for label, name in self.VECTOR_INDEXES.items():
            if name in have:
                status["vector"][label] = "ok"
                continue
            vt = self.vec_table(label)
            if "embedding" not in self.table_props(vt):
                status["vector"][label] = "missing column"
                continue
            if on_progress:
                on_progress(f"vector index {label}")
            try:
                self.execute(f"CALL CREATE_VECTOR_INDEX('{vt}', '{name}', 'embedding', "
                             f"metric := 'cosine')")
                status["vector"][label] = "ok"
            except Exception as exc:  # noqa: BLE001 - search falls back to brute force
                status["vector"][label] = f"error: {exc}"
        for label, (name, _props) in self.FTS_INDEXES.items():
            if name in have:            # a v4 database: drop it, see FTS_INDEXES
                try:
                    self.execute(f"CALL DROP_FTS_INDEX('{label}', '{name}')")
                except Exception:
                    pass
            status["fts"][label] = "kwindex"
        return status

    def drop_search_indexes(self) -> None:
        have = self.list_indexes()
        for label, name in self.VECTOR_INDEXES.items():
            if name in have:
                try:
                    self.execute(f"CALL DROP_VECTOR_INDEX('{have[name].get('table') or self.vec_table(label)}', '{name}')")
                except Exception:
                    pass
        for label, (name, _p) in self.FTS_INDEXES.items():
            if name in have:
                try:
                    self.execute(f"CALL DROP_FTS_INDEX('{label}', '{name}')")
                except Exception:
                    pass

    def vector_topk(self, label: str, vec, k: int, efs: int = 0) -> list[tuple[int, float]]:
        """[(id, cosine_similarity)] of the k nearest `label` rows via HNSW."""
        name = self.VECTOR_INDEXES[label]
        v = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        opt = f", efs := {int(efs)}" if efs else ""
        rows = self.fetch_all(
            f"CALL QUERY_VECTOR_INDEX('{self.vec_table(label)}', '{name}', $v, {int(k)}{opt}) "
            f"RETURN node.id AS id, distance AS d", {"v": v})
        return [(int(r["id"]), 1.0 - float(r["d"])) for r in rows]

    # ---- columnar reads / bulk updates for the graph analytics ------------
    # Everything below returns numpy arrays through Arrow, so a 1M-edge graph
    # never becomes a million Python dicts.

    SYMBOL_LABELS = ("File", "Class", "Function", "Variable")

    def _np(self, cypher: str, params: dict | None = None) -> dict[str, "np.ndarray"]:
        import numpy as np
        t = self.fetch_arrow(cypher, params)
        out: dict[str, np.ndarray] = {}
        for name in t.column_names:
            col = t.column(name)
            try:
                out[name] = col.to_numpy()
            except Exception:
                out[name] = np.asarray(col.to_pylist(), dtype=object)
        return out

    def edge_endpoints(self, rels, with_conf: bool = False, from_label: str | None = None,
                       to_label: str | None = None):
        """(src_ids, dst_ids, conf|None, rel_index) over the given rel tables."""
        import numpy as np
        srcs, dsts, confs, kinds = [], [], [], []
        fl = f":{from_label}" if from_label else ""
        tl = f":{to_label}" if to_label else ""
        for k, rel in enumerate(rels):
            if not self.has_table(rel):
                continue
            conf = "coalesce(r.confidence, 1.0)" if (with_conf and "confidence" in self.table_props(rel)) else "1.0"
            try:
                d = self._np(f"MATCH (a{fl})-[r:{rel}]->(b{tl}) RETURN a.id AS a, b.id AS b, "
                             f"CAST({conf} AS DOUBLE) AS c")
            except Exception:
                continue
            if "a" not in d or len(d["a"]) == 0:
                continue
            srcs.append(d["a"].astype(np.int64))
            dsts.append(d["b"].astype(np.int64))
            confs.append(d["c"].astype(np.float32))
            kinds.append(np.full(len(d["a"]), k, dtype=np.int16))
        if not srcs:
            e = np.zeros(0, np.int64)
            return e, e, (np.zeros(0, np.float32) if with_conf else None), np.zeros(0, np.int16)
        return (np.concatenate(srcs), np.concatenate(dsts),
                np.concatenate(confs) if with_conf else None, np.concatenate(kinds))

    def node_columns(self, label: str, cols: dict[str, str], where: str = "",
                     params: dict | None = None) -> dict[str, "np.ndarray"]:
        """{alias: array} for `RETURN <expr> AS <alias>` over one label."""
        ret = ", ".join(f"{expr} AS {alias}" for alias, expr in cols.items())
        w = f" WHERE {where}" if where else ""
        return self._np(f"MATCH (n:{label}){w} RETURN {ret}", params)

    def embedding_matrix(self, label: str):
        """(ids int64, float32 matrix) of every row that has an embedding."""
        import numpy as np
        t = self.fetch_arrow(f"MATCH (n:{self.vec_table(label)}) WHERE n.embedding IS NOT NULL "
                             f"RETURN n.id AS id, n.embedding AS e")
        ids = t.column("id").to_numpy().astype(np.int64)
        if len(ids) == 0:
            return ids, np.zeros((0, self.embedding_dim), np.float32)
        col = t.column("e").combine_chunks()
        mat = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32)
        return ids, mat.reshape(len(ids), -1)

    def embeddings_for(self, label: str, ids: list[int]):
        """(ids, matrix) for specific rows."""
        import numpy as np
        if not ids:
            return np.zeros(0, np.int64), np.zeros((0, self.embedding_dim), np.float32)
        # No `embedding IS NOT NULL` predicate: Kuzu would then read the
        # embedding column of every row of the scan, not just the matches.
        vt = self.vec_table(label)
        if len(ids) <= 2000:   # small sets: the row path beats Arrow's list conversion
            rows = [r for r in self.fetch_all(
                f"MATCH (n:{vt}) WHERE n.id IN $ids RETURN n.id AS id, n.embedding AS e",
                {"ids": [int(i) for i in ids]}) if r.get("e") is not None]
            if not rows:
                return np.zeros(0, np.int64), np.zeros((0, self.embedding_dim), np.float32)
            return (np.array([r["id"] for r in rows], dtype=np.int64),
                    np.array([r["e"] for r in rows], dtype=np.float32))
        t = self.fetch_arrow(f"MATCH (n:{vt}) WHERE n.id IN $ids "
                             f"RETURN n.id AS id, n.embedding AS e", {"ids": [int(i) for i in ids]})
        got = t.column("id").to_numpy().astype(np.int64)
        if len(got) == 0:
            return got, np.zeros((0, self.embedding_dim), np.float32)
        col = t.column("e").combine_chunks()
        valid = ~np.asarray(col.is_null().to_numpy(zero_copy_only=False), dtype=bool)
        if not valid.all():
            got = got[valid]
            col = col.filter(valid)
        if len(got) == 0:
            return got, np.zeros((0, self.embedding_dim), np.float32)
        mat = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.float32)
        return got, mat.reshape(len(got), -1)

    def set_node_values(self, label: str, ids, cols: dict[str, "np.ndarray"],
                        batch: int = 500_000) -> int:
        """Bulk `SET n.<col> = value` by primary key.

        `LOAD FROM <arrow> ... MATCH (n) WHERE n.id = i SET ...` plans as a
        hash join: 250k updates take ~0.3 s, where `UNWIND $rows MATCH (n
        {id: row.id}) SET` took ~9 s per 50k (one PK probe + Python->Kuzu
        conversion per row)."""
        import importlib.util  # noqa: F401 - kuzu's Arrow scan needs it imported
        import numpy as np
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return 0
        names = list(cols)
        sets = ", ".join(f"n.{c} = _v_{c}" for c in names)
        withs = ", ".join(["_i AS _i"] + [f"_v_{c} AS _v_{c}" for c in names])
        q = (f"LOAD FROM set_arrow WITH {withs} "
             f"MATCH (n:{label}) WHERE n.id = _i SET {sets}")
        vals = {c: np.asarray(v) for c, v in cols.items()}
        for s0 in range(0, len(ids), batch):
            sl = slice(s0, s0 + batch)
            data = {"_i": pa.array(ids[sl], pa.int64())}
            for c in names:
                v = vals[c][sl]
                if v.dtype.kind == "f":
                    data[f"_v_{c}"] = pa.array(v.astype(np.float64), pa.float64())
                elif v.dtype.kind in "iu":
                    data[f"_v_{c}"] = pa.array(v.astype(np.int64), pa.int64())
                elif v.dtype.kind == "b":
                    data[f"_v_{c}"] = pa.array(v, pa.bool_())
                else:
                    data[f"_v_{c}"] = pa.array([None if x is None else str(x) for x in v.tolist()],
                                               pa.string())
            set_arrow = pa.table(data)  # noqa: F841 - referenced by name in the query
            self.execute(q)
        return len(ids)

    def memberships(self):
        """(node_ids, community_ids) of every MEMBER_OF edge."""
        import numpy as np
        if not self.has_table("Community"):
            e = np.zeros(0, np.int64)
            return e, e
        try:
            d = self._np("MATCH (n)-[:MEMBER_OF]->(c:Community) RETURN n.id AS n, c.id AS c")
        except Exception:
            e = np.zeros(0, np.int64)
            return e, e
        if "n" not in d:
            e = np.zeros(0, np.int64)
            return e, e
        return d["n"].astype(np.int64), d["c"].astype(np.int64)

    def community_rows(self) -> list[dict]:
        if not self.has_table("Community"):
            return []
        cols = "c.id AS id, c.name AS name, c.size AS size"
        if "x" in self.table_props("Community"):
            cols += ", c.x AS x, c.y AS y, c.r AS r"
        return self.fetch_all(f"MATCH (c:Community) RETURN {cols}")

    def layout_source(self):
        """Arrays for layout / tiles: every File/Class/Function/Variable with
        kind, file path, line, pagerank, position, name, test flag."""
        import numpy as np
        parts = []
        kind_of = {"File": 0, "Class": 1, "Function": 2, "Variable": 3}
        for label in self.SYMBOL_LABELS:
            has_xy = "x" in self.table_props(label)
            xy = "n.x AS x, n.y AS y" if has_xy else "CAST(NULL AS DOUBLE) AS x, CAST(NULL AS DOUBLE) AS y"
            if label == "File":
                cols = (f"n.id AS id, n.path AS file, n.path AS name, 0 AS line, "
                        f"coalesce(n.pagerank, 0.0) AS pr, false AS t, {xy}")
            elif label == "Variable":
                cols = (f"n.id AS id, n.file AS file, n.name AS name, coalesce(n.line, 0) AS line, "
                        f"0.0 AS pr, false AS t, {xy}")
            else:
                t = "coalesce(n.is_test, false)" if label == "Function" else "false"
                cols = (f"n.id AS id, n.file AS file, n.name AS name, coalesce(n.line_start, 0) AS line, "
                        f"coalesce(n.pagerank, 0.0) AS pr, {t} AS t, {xy}")
            tb = self.fetch_arrow(f"MATCH (n:{label}) RETURN {cols}")
            if tb.num_rows == 0:
                continue
            parts.append((kind_of[label], tb))
        if not parts:
            e = np.zeros(0)
            return {"ids": e.astype(np.int64), "kinds": e.astype(np.uint8), "files": [],
                    "names": [], "line": e, "pr": e, "test": e.astype(bool), "x": e, "y": e}
        ids = np.concatenate([p[1].column("id").to_numpy().astype(np.int64) for p in parts])
        kinds = np.concatenate([np.full(p[1].num_rows, p[0], np.uint8) for p in parts])
        files: list[str] = []
        names: list[str] = []
        for _k, tb in parts:
            files.extend(str(v or "") for v in tb.column("file").to_pylist())
            names.extend(str(v or "") for v in tb.column("name").to_pylist())

        def num(col: str, dtype) -> np.ndarray:
            out = []
            for _k, tb in parts:
                c = tb.column(col).combine_chunks()
                if c.null_count:
                    c = c.fill_null(float("nan")) if str(c.type) != "bool" else c.fill_null(False)
                out.append(np.asarray(c.to_numpy(zero_copy_only=False), dtype=np.float64))
            return np.concatenate(out).astype(dtype)
        return {"ids": ids, "kinds": kinds, "files": files, "names": names,
                "line": num("line", np.float64), "pr": num("pr", np.float64),
                "test": num("t", np.float64) > 0, "x": num("x", np.float64), "y": num("y", np.float64)}

    # ---- live-index helpers (all by primary key / id lists) ---------------

    def symbol_rows(self) -> list[tuple[str, int, str, str, str]]:
        """[(label, id, name, qname, file)] of every Function / Class /
        Variable, in id order (= parse order)."""
        out: list[tuple[str, int, str, str, str]] = []
        for label in ("Function", "Class", "Variable"):
            t = self.fetch_arrow(f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, "
                                 f"n.qname AS q, n.file AS f")
            if t.num_rows == 0:
                continue
            import sys as _sys
            ids = t.column("id").to_pylist()
            # names and paths repeat (a path once per entity, common names
            # thousands of times): intern them so the live symbol table holds
            # one string object each
            names = [_sys.intern(x or "") for x in t.column("name").to_pylist()]
            qs = t.column("q").to_pylist()
            fs = [_sys.intern(x or "") for x in t.column("f").to_pylist()]
            out.extend(zip([label] * len(ids), ids, names, qs, fs))
        out.sort(key=lambda r: r[1])
        return out

    def file_ids(self) -> dict[str, int]:
        t = self.fetch_arrow("MATCH (f:File) RETURN f.path AS p, f.id AS id")
        return dict(zip(t.column("p").to_pylist(), t.column("id").to_pylist()))

    def module_rows(self) -> list[dict]:
        return self.fetch_all("MATCH (m:Module) RETURN m.id AS id, m.name AS name")

    def insert_vectors(self, label: str, vec_rows: list[dict]) -> int:
        """(id, embedding) rows into `label`'s vector side table."""
        vt = self.VEC_TABLES.get(label)
        if not vec_rows or not vt or not self.has_table(vt):
            return 0
        return self.insert_nodes(vt, vec_rows)

    def delete_by_ids(self, label: str, ids: list[int]) -> None:
        if ids and self.journal is not None:
            self.journal.ops.append(("del_nodes", [int(i) for i in ids]))
        vt = self.VEC_TABLES.get(label)
        if ids and vt and self.has_table(vt):
            for s0 in range(0, len(ids), 20_000):
                self.execute(f"MATCH (n:{vt}) WHERE n.id IN $ids DELETE n",
                             {"ids": [int(i) for i in ids[s0:s0 + 20_000]]})
        for s0 in range(0, len(ids), 20_000):
            self.execute(f"MATCH (n:{label}) WHERE n.id IN $ids DETACH DELETE n",
                         {"ids": [int(i) for i in ids[s0:s0 + 20_000]]})

    def ids_in_files(self, label: str, files: list[str]) -> list[int]:
        if not files or not self.has_table(label):
            return []
        prop = "path" if label == "File" else "file"
        rows = self.fetch_all(f"MATCH (n:{label}) WHERE n.{prop} IN $f RETURN n.id AS id", {"f": files})
        return [int(r["id"]) for r in rows]

    def embeddings_by_ids(self, label: str, ids: list[int]) -> dict[str, list]:
        """{ehash: embedding} for rows by id -- the pre-delete harvest."""
        if not ids or "ehash" not in self.table_props(label):
            return {}
        return self._vectors_by_hash_rows(label, [
            (int(r["id"]), r["h"]) for s0 in range(0, len(ids), 5000)
            for r in self.fetch_all(f"MATCH (n:{label}) WHERE n.id IN $ids RETURN n.id AS id, n.ehash AS h",
                                    {"ids": [int(i) for i in ids[s0:s0 + 5000]]}) if r.get("h")])

    def _vectors_by_hash_rows(self, label: str, id_hash: list[tuple[int, str]]) -> dict[str, list]:
        """{ehash: embedding} for (id, ehash) pairs, vectors read by id."""
        if not id_hash:
            return {}
        got, mat = self.embeddings_for(label, [i for i, _h in id_hash])
        vec_of = {int(i): mat[j] for j, i in enumerate(got.tolist())}
        out: dict[str, list] = {}
        for i, h in id_hash:
            v = vec_of.get(int(i))
            if v is not None and h not in out:
                out[h] = v
        return out

    def layout_by_ids(self, label: str, ids: list[int]) -> dict[int, tuple[float, float, float]]:
        """{id: (x, y, pagerank)} for rows that have a position."""
        if not ids or "x" not in self.table_props(label):
            return {}
        pr = "coalesce(n.pagerank, 0.0)" if label != "Variable" else "0.0"
        out: dict[int, tuple[float, float, float]] = {}
        for s0 in range(0, len(ids), 20_000):
            for r in self.fetch_all(f"MATCH (n:{label}) WHERE n.id IN $ids AND n.x IS NOT NULL "
                                    f"RETURN n.id AS id, n.x AS x, n.y AS y, {pr} AS pr",
                                    {"ids": [int(i) for i in ids[s0:s0 + 20_000]]}):
                out[int(r["id"])] = (float(r["x"]), float(r["y"]), float(r["pr"] or 0.0))
        return out

    def communities_of(self, ids: list[int]) -> dict[int, int]:
        if not ids or not self.has_table("Community"):
            return {}
        rows = self.fetch_all("MATCH (n)-[:MEMBER_OF]->(c:Community) WHERE n.id IN $ids "
                              "RETURN n.id AS n, c.id AS c", {"ids": [int(i) for i in ids]})
        return {int(r["n"]): int(r["c"]) for r in rows}

    # (rel, from_label, to_label) pairs a source entity resolves out of
    RESOLVED_OUT = (("CALLS", "Function", "Function"), ("CALLS_CANDIDATE", "Function", "Function"),
                    ("INSTANTIATES", "Function", "Class"), ("INHERITS", "Class", "Class"),
                    ("DECORATED_BY", "Function", "Function"), ("DECORATED_BY", "Class", "Function"),
                    ("IMPORTS_SYMBOL", "File", "Class"), ("IMPORTS_SYMBOL", "File", "Function"),
                    ("HANDLES", "Route", "Function"), ("HANDLES", "Tool", "Function"))

    # (rel, from_label, to_label, props) whose target may sit in a changed
    # file: incoming edges an incremental pass re-points at the recreated
    # target instead of re-resolving every caller (see Indexer._resolve_scope)
    REMAP_IN = (("CALLS", "Function", "Function", ("line", "confidence", "method")),
                ("CALLS_CANDIDATE", "Function", "Function", ("line", "confidence", "method")),
                ("INSTANTIATES", "Function", "Class", ("line", "confidence", "method")),
                ("INHERITS", "Class", "Class", ("confidence", "method")),
                ("DECORATED_BY", "Function", "Function", ()), ("DECORATED_BY", "Class", "Function", ()),
                ("IMPORTS_SYMBOL", "File", "Class", ()), ("IMPORTS_SYMBOL", "File", "Function", ()),
                ("HANDLES", "Route", "Function", ()), ("HANDLES", "Tool", "Function", ()),
                ("IMPORTS", "File", "File", ()))

    def incoming_edges(self, ids_by_label: dict[str, list[int]]) -> list[tuple]:
        """[(rel, fl, tl, src_id, src_file, dst_id, props)] of the REMAP_IN
        edges ending at the given nodes."""
        out = []
        for rel, fl, tl, props in self.REMAP_IN:
            ids = ids_by_label.get(tl)
            if not ids or not self.has_table(rel) or not self.has_table(fl):
                continue
            have = self.table_props(rel)
            pr = [p for p in props if p in have]
            fcol = "a.path" if fl == "File" else "a.file"
            ret = ", ".join(["a.id AS a", f"{fcol} AS f", "b.id AS b"] + [f"r.{p} AS {p}" for p in pr])
            for s0 in range(0, len(ids), 20_000):
                for r in self.fetch_all(f"MATCH (a:{fl})-[r:{rel}]->(b:{tl}) WHERE b.id IN $ids RETURN {ret}",
                                        {"ids": [int(i) for i in ids[s0:s0 + 20_000]]}):
                    out.append((rel, fl, tl, int(r["a"]), r["f"] or "", int(r["b"]),
                                {p: r.get(p) for p in pr}))
        return out

    def resolved_out_edges(self, ids_by_label: dict[str, list[int]]) -> list[tuple[str, str, str, int, int, str]]:
        """[(rel, from_label, to_label, src, dst, dst_name)] leaving the given nodes."""
        out = []
        for rel, fl, tl in self.RESOLVED_OUT:
            ids = ids_by_label.get(fl)
            if not ids or not self.has_table(rel) or not self.has_table(fl):
                continue
            for s0 in range(0, len(ids), 20_000):
                for r in self.fetch_all(
                        f"MATCH (a:{fl})-[r:{rel}]->(b:{tl}) WHERE a.id IN $ids "
                        f"RETURN a.id AS a, b.id AS b, b.name AS n",
                        {"ids": [int(i) for i in ids[s0:s0 + 20_000]]}):
                    out.append((rel, fl, tl, int(r["a"]), int(r["b"]), r["n"] or ""))
        return out

    def delete_edge_pairs(self, rel: str, fl: str, tl: str, pairs: list[tuple[int, int]]) -> None:
        import importlib.util  # noqa: F401
        if not pairs:
            return
        if self.journal is not None:
            self.journal.ops.append(("del_pairs", rel, [int(a) for a, _ in pairs],
                                     [int(b) for _, b in pairs]))
        del_arrow = pa.table({"_a": pa.array([int(a) for a, _ in pairs], pa.int64()),  # noqa: F841
                              "_b": pa.array([int(b) for _, b in pairs], pa.int64())})
        self.execute(f"LOAD FROM del_arrow WITH _a AS _a, _b AS _b "
                     f"MATCH (a:{fl})-[r:{rel}]->(b:{tl}) WHERE a.id = _a AND b.id = _b DELETE r")

    def delete_out_edges(self, rel: str, fl: str, tl: str, ids: list[int]) -> None:
        if not ids or not self.has_table(rel):
            return
        if self.journal is not None:
            self.journal.ops.append(("del_out", rel, [int(i) for i in ids]))
        for s0 in range(0, len(ids), 20_000):
            self.execute(f"MATCH (a:{fl})-[r:{rel}]->(b:{tl}) WHERE a.id IN $ids DELETE r",
                         {"ids": [int(i) for i in ids[s0:s0 + 20_000]]})

    def delete_all_edges(self, rel: str) -> None:
        if not self.has_table(rel):
            return
        if self.journal is not None:
            self.journal.ops.append(("del_kind", rel))
        self.execute(f"MATCH ()-[r:{rel}]->() DELETE r")

    def delete_edges_touching(self, rel: str, fl: str, tl: str, ids: list[int]) -> None:
        """Edges of `rel` with either endpoint in `ids` (journalled as the
        two directions)."""
        if not ids or not self.has_table(rel):
            return
        ids = [int(i) for i in ids]
        rows = self.fetch_all(f"MATCH (a:{fl})-[r:{rel}]->(b:{tl}) WHERE a.id IN $ids OR b.id IN $ids "
                              f"RETURN a.id AS a, b.id AS b", {"ids": ids})
        if rows:
            self.delete_edge_pairs(rel, fl, tl, [(int(r["a"]), int(r["b"])) for r in rows])

    def inherits_pairs(self) -> list[tuple[int, int]]:
        if not self.has_table("INHERITS"):
            return []
        t = self.fetch_arrow("MATCH (a:Class)-[:INHERITS]->(b:Class) RETURN a.id AS a, b.id AS b")
        return list(zip(t.column("a").to_pylist(), t.column("b").to_pylist()))

    def test_function_rows(self) -> list[dict]:
        """[{id, name}] of every test function."""
        return self.fetch_all("MATCH (n:Function) WHERE n.is_test RETURN n.id AS id, n.name AS name")

    def node_lines(self, ids_by_label: dict[str, list[int]]) -> dict[int, int]:
        """{id: first line} for Class / Function / Variable rows by id."""
        out: dict[int, int] = {}
        for label, col in (("Class", "line_start"), ("Function", "line_start"), ("Variable", "line")):
            ids = ids_by_label.get(label) or []
            for s0 in range(0, len(ids), 20_000):
                for r in self.fetch_all(f"MATCH (n:{label}) WHERE n.id IN $ids "
                                        f"RETURN n.id AS id, coalesce(n.{col}, 0) AS l",
                                        {"ids": [int(i) for i in ids[s0:s0 + 20_000]]}):
                    out[int(r["id"])] = int(r["l"] or 0)
        return out

    def framework_nodes_in(self, files: list[str]) -> dict[tuple[str, int, str], tuple[str, int]]:
        out: dict[tuple[str, int, str], tuple[str, int]] = {}
        if not files:
            return out
        for label in ("Route", "Tool"):
            if not self.has_table(label):
                continue
            for r in self.fetch_all(f"MATCH (n:{label}) WHERE n.file IN $f "
                                    f"RETURN n.id AS id, n.file AS file, n.line AS line, n.name AS name",
                                    {"f": files}):
                out[(r["file"], int(r["line"] or 0), r["name"])] = (label, int(r["id"]))
        return out

    def test_functions(self, ids: list[int]) -> list[dict]:
        if not ids:
            return []
        return self.fetch_all("MATCH (n:Function) WHERE n.id IN $ids AND n.is_test "
                              "RETURN n.id AS id, n.name AS name, n.is_test AS is_test",
                              {"ids": [int(i) for i in ids]})

    # ---- small targeted helpers for the incremental analytics path ------

    def count_symbols(self, files: list[str] | None = None) -> int:
        """File + Class + Function + Variable nodes (optionally in `files`)."""
        total = 0
        for label in self.SYMBOL_LABELS:
            prop = "path" if label == "File" else "file"
            where = f" WHERE n.{prop} IN $f" if files is not None else ""
            try:
                r = self.fetch_all(f"MATCH (n:{label}){where} RETURN count(n) AS c",
                                   {"f": files} if files is not None else None)
                total += int(r[0]["c"]) if r else 0
            except Exception:
                pass
        return total

    def missing_positions(self) -> int:
        """Files without a layout position (0 when the column is missing
        counts as 'all missing')."""
        if "x" not in self.table_props("File"):
            return 1 << 30
        r = self.fetch_all("MATCH (f:File) WHERE f.x IS NULL RETURN count(f) AS c")
        return int(r[0]["c"]) if r else 0

    _NEIGHBOR_RELS = "CALLS|CONTAINS|IMPORTS|IMPORTS_SYMBOL|INHERITS|INSTANTIATES"

    def neighbor_communities(self, ids: list[int]) -> list[tuple[int, int]]:
        if not ids or not self.has_table("Community"):
            return []
        rows = self.fetch_all(
            f"MATCH (a)-[:{self._NEIGHBOR_RELS}]-(b)-[:MEMBER_OF]->(c:Community) "
            f"WHERE a.id IN $ids RETURN a.id AS a, c.id AS c", {"ids": [int(i) for i in ids]})
        return [(int(r["a"]), int(r["c"])) for r in rows]

    def neighbor_center(self, ids: list[int]) -> tuple[float, float] | None:
        """Mean position of already-placed neighbours of `ids` outside them."""
        if not ids:
            return None
        rows = self.fetch_all(
            f"MATCH (a)-[:{self._NEIGHBOR_RELS}]-(b) WHERE a.id IN $ids AND NOT b.id IN $ids "
            f"AND b.x IS NOT NULL RETURN avg(b.x) AS x, avg(b.y) AS y, count(b) AS n",
            {"ids": [int(i) for i in ids]})
        if rows and rows[0].get("n") and rows[0].get("x") is not None:
            return float(rows[0]["x"]), float(rows[0]["y"])
        return None

    def community_center_of(self, file_id: int) -> tuple[float, float] | None:
        if not self.has_table("Community") or "x" not in self.table_props("Community"):
            return None
        rows = self.fetch_all("MATCH (f:File)-[:MEMBER_OF]->(c:Community) WHERE f.id = $id "
                              "AND c.x IS NOT NULL RETURN c.x AS x, c.y AS y", {"id": int(file_id)})
        return (float(rows[0]["x"]), float(rows[0]["y"])) if rows else None

    def layout_extent(self) -> float:
        if "x" not in self.table_props("File"):
            return 0.0
        rows = self.fetch_all("MATCH (f:File) WHERE f.x IS NOT NULL "
                              "RETURN max(abs(f.x)) AS a, max(abs(f.y)) AS b")
        if not rows or rows[0].get("a") is None:
            return 0.0
        return float(max(rows[0]["a"], rows[0]["b"]))

    def fetch_arrow(self, cypher: str, params: dict | None = None, chunk: int = 1_000_000):
        """Query result as one pyarrow Table (columnar; embeddings come back
        as a fixed-size-list column that converts to numpy without Python
        float objects)."""
        result = self.execute(cypher, params)
        return result.get_as_arrow(chunk)

    @contextmanager
    def thread_conn(self):
        """Route this thread's queries to its own connection on the same
        Database (Kuzu runs queries of different connections concurrently;
        a single connection must never be shared between threads). Used by
        the search fan-out; every helper method works unchanged inside."""
        if self.db is None:
            raise DatabaseBusy(f"graph DB busy: {self.db_path} is closed")
        self._thread_connection()
        prev = getattr(self._tls, "active", False)
        self._tls.active = True
        try:
            yield self
        finally:
            self._tls.active = prev

    def _thread_connection(self):
        c = getattr(self._tls, "conn", None)
        if c is None:
            c = kuzu.Connection(self.db)
            with self._side_lock:
                self._side.append(c)
            self._tls.conn = c
        return c

    # Kuzu 0.11 uses the primary-key index only for `x.id = <literal>`: a
    # parameter (`= $id`) or a list (`IN $ids`) scans the whole table (~7 ms
    # on 200k rows, ~40 ms for a 1000-id list). A read query whose only use of
    # the id list is `x.id IN $ids` is therefore rewritten, for short lists,
    # into a UNION ALL of literal equalities (11 ids: 2 ms instead of 8).
    PK_UNION_MAX = 48
    _PK_BAD = (" ORDER BY", " LIMIT", " SKIP", " WITH ", "UNWIND", " OR ", "EXISTS", "COUNT(",
               "COLLECT(", "DISTINCT", "SUM(", "AVG(", "MIN(", "MAX(", " DELETE", " SET ", "CREATE",
               "MERGE", "UNION", "CALL ", "LOAD ", "COPY ", "OPTIONAL")

    def _pk_rewrite(self, cypher: str, params: dict | None):
        if not params:
            return None
        if "id" in params and len(params) >= 1 and isinstance(params.get("id"), int) \
                and not isinstance(params.get("id"), bool) and cypher.count("$id") == 1:
            m = _PK_EQ.search(cypher)
            up = cypher.upper()
            write = any(b in up for b in (" DELETE", " SET ", "CREATE", "MERGE", "COPY ", "LOAD "))
            if m and not write and re.search(rf"MATCH \({m.group(1)}:\w+\)", cypher):
                return (cypher[:m.start()] + f"{m.group(1)}.id = {int(params['id'])}" + cypher[m.end():],
                        {k: v for k, v in params.items() if k != "id"})
        ids = params.get("ids")
        if not isinstance(ids, list) or not (0 < len(ids) <= self.PK_UNION_MAX) or cypher.count("$ids") != 1:
            return None
        up = cypher.upper()
        if not up.lstrip().startswith("MATCH") or " RETURN " not in up or any(b in up for b in self._PK_BAD):
            return None
        # node lookups only: a relationship pattern per literal is a scan each
        if "-[" in cypher or "]-" in cypher or "," in cypher.split("WHERE", 1)[0]:
            return None
        m = _PK_IN.search(cypher)
        if not m or not re.search(rf"MATCH \({m.group(1)}:\w+\)", cypher):
            return None
        try:
            lits = [int(i) for i in ids if not isinstance(i, bool)]
        except (TypeError, ValueError):
            return None
        if len(lits) != len(ids):
            return None
        head, tail = cypher[:m.start()], cypher[m.end():]
        parts = [f"{head}{m.group(1)}.id = {v}{tail}" for v in dict.fromkeys(lits)]
        return " UNION ALL ".join(parts), {k: v for k, v in params.items() if k != "ids"}

    def execute(self, cypher: str, params: dict | None = None) -> Any:
        if self.conn is None:
            raise DatabaseBusy(
                f"graph DB busy: connection to {self.db_path} is closed "
                "(writer active — retry shortly)"
            )
        # A handle is shared by the event loop, worker threads (routes run in
        # the thread pool) and -- on the host -- the indexer: every thread
        # gets its own connection, so reads run concurrently and a
        # connection is never used by two threads at once.
        rw = self._pk_rewrite(cypher, params)
        if rw is not None:
            cypher, params = rw
        return self._thread_connection().execute(cypher, params or {})

    def fetch_all(self, cypher: str, params: dict | None = None) -> list[dict]:
        result = self.execute(cypher, params)
        out: list[dict] = []
        cols = None
        while result.has_next():
            row = result.get_next()
            if cols is None:
                cols = result.get_column_names()
            out.append(dict(zip(cols, row)))
        return out

    @contextmanager
    def bulk(self):
        """Context manager for bulk write sessions. Currently a no-op; reserved
        for future Kuzu COPY-from-arrow optimizations."""
        try:
            yield self
        finally:
            pass

    def insert_nodes(
        self,
        table: str,
        rows: Iterable[dict],
        batch_size: int = 5000,
        on_progress: "Callable[[int], None] | None" = None,
    ) -> int:
        """Bulk insert via UNWIND, batched.

        Splits the input into chunks of `batch_size` so Kuzu materializes one
        slab at a time instead of the whole list. Caller can pass numpy
        ndarrays as embedding values — we convert just-in-time per batch
        (numpy float32 → Python list[float]) so the caller's row dicts can
        keep the cheap numpy form throughout their lifetime.
        """
        rows = list(rows)
        if not rows:
            return 0
        vt = self.VEC_TABLES.get(table)
        if vt and "embedding" in rows[0] and self.has_table(vt):
            # schema v5: the vector goes to the side table (now, or through
            # the host's vector_sink in the background)
            vec_rows = [{"id": r["id"], "embedding": r.pop("embedding")} for r in rows
                        if r.get("embedding") is not None]
            for r in rows:
                r.pop("embedding", None)
            n = self.insert_nodes(table, rows, batch_size=batch_size, on_progress=on_progress)
            if vec_rows:
                if self.vector_sink is not None:
                    self.vector_sink(table, vec_rows)
                else:
                    self.insert_vectors(table, vec_rows)
            return n
        # Bulk path: COPY FROM Arrow is ~75x faster than UNWIND for rows with
        # an embedding (UNWIND converts every float of every row). Only for
        # tables without a search index -- Kuzu's FTS index does not see
        # COPY'd rows, so indexed tables (incremental passes) take UNWIND.
        if len(rows) >= self.COPY_MIN_ROWS:
            try:
                return self._copy_nodes(table, rows, batch_size=max(batch_size, 20_000),
                                        on_progress=on_progress)
            except Exception as exc:  # noqa: BLE001 - fall back to the portable path
                import logging
                logging.getLogger(__name__).debug("COPY %s failed, using UNWIND: %s", table, exc)
        keys = list(rows[0].keys())
        cols = ", ".join(f"{k}: row.{k}" for k in keys)
        cypher = f"UNWIND $rows AS row CREATE (n:{table} {{{cols}}})"
        n = len(rows)
        # Extend the id cache if it's already populated for this table —
        # otherwise leave it untouched and let `_ensure_known_ids` lazy-load.
        cached = self._known_ids.get(table)
        for i in range(0, n, batch_size):
            slab = rows[i : i + batch_size]
            for r in slab:
                v = r.get("embedding")
                if v is not None and not isinstance(v, list):
                    # numpy / array-like → list[float] just for the wire call
                    r["embedding"] = v.tolist() if hasattr(v, "tolist") else list(v)
            self.execute(cypher, {"rows": slab})
            if cached is not None:
                for r in slab:
                    cached.add(r["id"])
            if on_progress is not None:
                on_progress(len(slab))
        return n

    # COPY FROM has a fixed cost that grows with the table (measured on a
    # 200k-row Function table with an HNSW index: 12 rows via COPY 260-500 ms,
    # via UNWIND CREATE ~40 ms); bulk loads take COPY, small batches UNWIND.
    COPY_MIN_ROWS = 2000
    # Same for relationships: up to this many rows go through
    # LOAD FROM <arrow> + two PK MATCHes + CREATE (~10 ms for 10 rows where a
    # COPY costs 30-50 ms per rel table); dangling ids simply match nothing,
    # so no existence pre-check is needed either.
    SMALL_EDGE_ROWS = 5000

    def _has_search_index(self, table: str) -> bool:
        have = self.list_indexes()
        return any(v.get("table") == table for v in have.values())

    def _arrow_type(self, ktype: str):
        t = ktype.upper()
        if t.startswith("FLOAT[") or t.startswith("DOUBLE["):
            return None  # fixed-size list, built separately
        return {"INT64": pa.int64(), "INT32": pa.int32(), "STRING": pa.string(),
                "DOUBLE": pa.float64(), "FLOAT": pa.float32(), "BOOL": pa.bool_(),
                "BOOLEAN": pa.bool_()}.get(t, pa.string())

    def _copy_nodes(self, table: str, rows: list[dict], batch_size: int = 20_000,
                    on_progress: "Callable[[int], None] | None" = None) -> int:
        import importlib.util  # noqa: F401 - kuzu's Arrow scan uses it without importing
        import numpy as np
        types = self._col_types.get(table)
        if types is None:
            types = {}
            for r in self.fetch_all(f"CALL table_info('{table}') RETURN *"):
                types[str(r.get("name"))] = str(r.get("type"))
            self._col_types[table] = types
        keys = [k for k in rows[0].keys() if k in types]
        cached = self._known_ids.get(table)
        for i in range(0, len(rows), batch_size):
            slab = rows[i:i + batch_size]
            cols = {}
            for k in keys:
                kt = types[k]
                if "[" in kt:
                    dim = int(kt[kt.index("[") + 1:-1])
                    dt = np.float32 if kt.upper().startswith("FLOAT") else np.float64
                    mat = np.empty((len(slab), dim), dtype=dt)
                    mask = np.zeros(len(slab), dtype=bool)
                    for j, r in enumerate(slab):
                        v = r.get(k)
                        if v is None:
                            mask[j] = True
                            mat[j] = 0
                        else:
                            mat[j] = np.asarray(v, dtype=dt)
                    arr = pa.FixedSizeListArray.from_arrays(pa.array(mat.reshape(-1)), dim)
                    if mask.any():
                        arr = pa.FixedSizeListArray.from_arrays(pa.array(mat.reshape(-1)), dim,
                                                                mask=pa.array(mask))
                    cols[k] = arr
                else:
                    cols[k] = pa.array([r.get(k) for r in slab], type=self._arrow_type(kt))
            nodes_arrow = pa.table(cols)
            self.execute(f"COPY {table}({', '.join(keys)}) FROM nodes_arrow")
            if cached is not None:
                for r in slab:
                    cached.add(r["id"])
            if on_progress is not None:
                on_progress(len(slab))
        return len(rows)

    def _ensure_known_ids(self, table: str) -> set[int]:
        """Lazy-load the set of existing primary-key ids for a node table.
        Cached on the instance; mutated by `insert_nodes` after first load."""
        ids = self._known_ids.get(table)
        if ids is None:
            try:
                col = self.fetch_arrow(f"MATCH (n:{table}) RETURN n.id AS id").column("id")
                ids = set(col.to_numpy().tolist())
            except Exception:
                ids = {r["id"] for r in self.fetch_all(f"MATCH (n:{table}) RETURN n.id AS id")}
            self._known_ids[table] = ids
        return ids

    # Up to this many distinct endpoint ids, existence is checked with a
    # targeted `id IN $ids` query instead of loading the table's whole id
    # set -- keeps small incremental inserts O(changed).
    TARGETED_ID_CHECK = 20_000

    def _existing_ids(self, table: str, want: set[int]) -> set[int]:
        cached = self._known_ids.get(table)
        if cached is not None:
            return cached
        if len(want) > self.TARGETED_ID_CHECK:
            return self._ensure_known_ids(table)
        found: set[int] = set()
        lst = list(want)
        for s in range(0, len(lst), 5000):
            for r in self.fetch_all(f"MATCH (n:{table}) WHERE n.id IN $ids RETURN n.id AS id",
                                    {"ids": lst[s:s + 5000]}):
                found.add(r["id"])
        return found
    def insert_edges(
        self,
        edge: str,
        from_table: str,
        to_table: str,
        rows: Iterable[dict],
        batch_size: int = 100_000,
        on_progress: "Callable[[int], None] | None" = None,
        validate: bool = True,
    ) -> int:
        """Bulk edge insert via Kuzu's COPY FROM (Arrow path).

        validate=False skips the endpoint-existence check -- for callers whose
        ids come from the live symbol table (every id is known to exist).

        batch_size=10_000 picks a sweet spot: small enough that the progress
        bar ticks visibly on large repos (a 500k-edge insert gets 50 updates
        instead of 5), large enough that the per-COPY setup overhead stays
        amortized. Override via the kwarg if profiling says otherwise.

        Stages each batch as a pyarrow Table with `from`, `to`, and any
        edge-property columns, then issues:

            COPY <edge> FROM <arrow_var> (from='<FromTable>', to='<ToTable>')

        Kuzu's bulk loader uses the PK index for both endpoints — typically
        10-50× faster than the per-row MATCH+CREATE pattern at scale. The
        `(from=, to=)` clause is required because rel tables can declare
        multiple `(FROM, TO)` pairs (see EDGE_DDL).

        Dangling endpoints: the old MATCH+CREATE silently dropped rows whose
        `from_id`/`to_id` didn't resolve. COPY FROM aborts the batch on a
        missing PK, so we filter rows against `_known_ids[<table>]` before
        staging — preserving the old "best-effort, tolerant" behavior.
        """
        rows = list(rows)
        if not rows:
            return 0
        if len(rows) <= self.SMALL_EDGE_ROWS:
            prop_keys = [k for k in rows[0].keys() if k not in ("from_id", "to_id")]
            self._load_edges(edge, from_table, to_table, rows, prop_keys)
            if on_progress is not None:
                on_progress(len(rows))
            return len(rows)

        if validate:
            from_ids = self._existing_ids(from_table, {r["from_id"] for r in rows})
            to_ids = self._existing_ids(to_table, {r["to_id"] for r in rows})
            valid = [
                r for r in rows
                if r["from_id"] in from_ids and r["to_id"] in to_ids
            ]
        else:
            valid = rows
        if not valid:
            return 0

        prop_keys = [k for k in valid[0].keys() if k not in ("from_id", "to_id")]
        n = len(valid)
        if not validate:
            try:
                self._copy_edges(edge, from_table, to_table, valid, prop_keys, batch_size, on_progress)
            except Exception as exc:  # noqa: BLE001 - a dangling id: take the checked path
                import logging
                logging.getLogger(__name__).warning("unchecked COPY %s failed (%s); validating", edge, exc)
                return self.insert_edges(edge, from_table, to_table, valid, batch_size,
                                         on_progress, validate=True)
            return n
        self._copy_edges(edge, from_table, to_table, valid, prop_keys, batch_size, on_progress)
        return n

    def _load_edges(self, edge: str, fl: str, tl: str, rows: list[dict], prop_keys: list[str]) -> None:
        """Small relationship batch: LOAD FROM an Arrow table, look both
        endpoints up by primary key, CREATE. Rows whose endpoints do not
        exist produce nothing (the MATCH is empty)."""
        import importlib.util  # noqa: F401 - kuzu's Arrow scan needs it imported
        cols = {"_a": pa.array([int(r["from_id"]) for r in rows], pa.int64()),
                "_b": pa.array([int(r["to_id"]) for r in rows], pa.int64())}
        keys = []
        for k in prop_keys:
            vals = [r.get(k) for r in rows]
            if all(v is None for v in vals):
                continue
            if all(isinstance(v, (int, bool)) or v is None for v in vals) and not any(isinstance(v, bool) for v in vals):
                arr = pa.array(vals, pa.int64())
            elif all(isinstance(v, (int, float)) or v is None for v in vals):
                arr = pa.array([None if v is None else float(v) for v in vals], pa.float64())
            else:
                arr = pa.array([None if v is None else str(v) for v in vals], pa.string())
            cols[f"_p_{k}"] = arr
            keys.append(k)
        load_arrow = pa.table(cols)  # noqa: F841 - referenced by name in the query
        withs = ", ".join(f"{c} AS {c}" for c in cols)
        carry = ", ".join(["x", "_b"] + [f"_p_{k}" for k in keys])
        props = (" {" + ", ".join(f"{k}: _p_{k}" for k in keys) + "}") if keys else ""
        self.execute(f"LOAD FROM load_arrow WITH {withs} MATCH (x:{fl}) WHERE x.id = _a "
                     f"WITH {carry} MATCH (y:{tl}) WHERE y.id = _b CREATE (x)-[:{edge}{props}]->(y)")
        if self.journal is not None:
            conf = [r.get("confidence") for r in rows] if "confidence" in keys else None
            self.journal.ops.append(("add", edge, [int(r["from_id"]) for r in rows],
                                     [int(r["to_id"]) for r in rows], conf))

    def _copy_edges(self, edge, from_table, to_table, valid, prop_keys, batch_size, on_progress):
        n = len(valid)
        for i in range(0, n, batch_size):
            slab = valid[i : i + batch_size]
            cols: dict[str, list] = {
                "from": [r["from_id"] for r in slab],
                "to": [r["to_id"] for r in slab],
            }
            for k in prop_keys:
                cols[k] = [r[k] for r in slab]
            arrow = pa.table(cols)
            self.execute(
                f"COPY {edge} FROM arrow (from='{from_table}', to='{to_table}')"
            )
            if self.journal is not None:
                conf = cols.get("confidence") or cols.get("score")
                self.journal.ops.append(("add", edge, cols["from"], cols["to"],
                                         conf if edge != "SIMILAR_TO" and conf is not None else None))
            if on_progress is not None:
                on_progress(len(slab))

    def close(self) -> None:
        """Explicitly close the Kuzu connection + database. Required after
        a write session so a subsequent `read_only=True` open can acquire
        the file lock — GC isn't reliable on Windows + COPY FROM holds extra
        internal references that survive a `del`."""
        for c in list(getattr(self, "_side", [])):
            try:
                c.close()
            except Exception:
                pass
        if getattr(self, "_side", None) is not None:
            self._side.clear()
        try:
            if self.conn is not None and not self.conn.is_closed:
                self.conn.close()
        except Exception:
            pass
        try:
            if self.db is not None and not self.db.is_closed:
                self.db.close()
        except Exception:
            pass
        self.conn = None
        self.db = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def wipe(db_path: Path) -> None:
        if db_path.exists():
            if db_path.is_dir():
                shutil.rmtree(db_path)
            else:
                db_path.unlink()
