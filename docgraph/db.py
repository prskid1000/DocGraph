"""Kuzu storage layer. One file per repo at .docgraph/graph.kuzu.

Schema covers all Tier 1-4 relationships.
"""
from __future__ import annotations

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
        embedding FLOAT[{dim}],
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
        embedding FLOAT[{dim}],
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
        embedding FLOAT[{dim}],
        ehash STRING,
        line_start INT64,
        line_end INT64,
        terms STRING,
        PRIMARY KEY (id)
    )""",
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
SCHEMA_VERSION = 4

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


class GraphDB:
    def __init__(self, db_path: Path, embedding_dim: int = 384, read_only: bool = False):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.db = kuzu.Database(str(self.db_path), read_only=read_only, max_db_size=MAX_DB_SIZE)
        self.conn = kuzu.Connection(self.db)
        # Per-node-table id sets, populated lazily on first edge insert. Used
        # to filter dangling-endpoint rows before COPY FROM (which errors hard
        # on unknown PKs, vs. the old MATCH+CREATE which silently dropped).
        # `insert_nodes` extends the cache so freshly inserted nodes are seen.
        self._known_ids: dict[str, set[int]] = {}
        self._props_cache: dict[str, set[str]] = {}
        self.read_only = read_only

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
        return bool(self.table_props(table))

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
        if not rows:
            return 0
        self.execute(
            "UNWIND $rows AS row "
            "MATCH (a:Function {id: row.a})-[r:CALLS]->(b:Function {id: row.b}) "
            "SET r.confidence = row.confidence, r.method = row.method",
            {"rows": rows},
        )
        return len(rows)

    def set_history(self, label: str, rows: list[dict]) -> None:
        """Update history columns in place: rows [{id, fc, fts, lc, lts}]."""
        if not rows:
            return
        self.execute(
            f"UNWIND $rows AS row MATCH (n:{label} {{id: row.id}}) "
            f"SET n.first_seen_commit = row.fc, n.first_seen_ts = row.fts, "
            f"n.last_changed_commit = row.lc, n.last_changed_ts = row.lts",
            {"rows": rows},
        )

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

    def embeddings_in_files(self, label: str, files: list[str]) -> dict[str, list]:
        """{ehash: embedding} for every node of `label` in `files` -- the
        pre-delete harvest of the embedding cache."""
        if not files or "ehash" not in self.table_props(label):
            return {}
        out: dict[str, list] = {}
        for r in self.fetch_all(
            f"MATCH (n:{label}) WHERE n.file IN $files AND n.ehash IS NOT NULL "
            f"RETURN n.ehash AS h, n.embedding AS e",
            {"files": files},
        ):
            if r.get("h") and r.get("e") is not None:
                out.setdefault(r["h"], r["e"])
        return out

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

    def framework_nodes(self) -> dict[tuple[str, int, str], tuple[str, int]]:
        """{(file, line, name): (label, id)} for existing Route/Tool nodes."""
        out: dict[tuple[str, int, str], tuple[str, int]] = {}
        for label in ("Route", "Tool"):
            if not self.has_table(label):
                continue
            for r in self.fetch_all(
                f"MATCH (n:{label}) RETURN n.id AS id, n.file AS file, n.line AS line, n.name AS name"
            ):
                out[(r["file"], int(r["line"] or 0), r["name"])] = (label, r["id"])
        return out

    def embeddings_by_ehash(self, label: str, hashes: list[str],
                            files: list[str] | None = None) -> dict[str, list]:
        """Existing vectors for content hashes (the embedding cache). With
        `files`, only nodes in those files are searched (the pre-delete
        harvest); otherwise the whole table."""
        if not hashes or "ehash" not in self.table_props(label):
            return {}
        out: dict[str, list] = {}
        if files is not None:
            q = (f"MATCH (n:{label}) WHERE n.file IN $files AND n.ehash IS NOT NULL "
                 f"RETURN n.ehash AS h, n.embedding AS e")
            params: dict = {"files": files}
        else:
            q = (f"MATCH (n:{label}) WHERE n.ehash IN $hs "
                 f"RETURN n.ehash AS h, n.embedding AS e")
            params = {"hs": hashes}
        want = set(hashes)
        for r in self.fetch_all(q, params):
            h = r.get("h")
            if h in want and h not in out and r.get("e") is not None:
                out[h] = r["e"]
        return out

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
        """Create any missing vector / FTS index. Returns a status dict
        {vector: {label: ok|error}, fts: {label: ok|error}}; never raises."""
        have = self.list_indexes()
        status: dict[str, dict[str, str]] = {"vector": {}, "fts": {}}
        for label, name in self.VECTOR_INDEXES.items():
            if name in have:
                status["vector"][label] = "ok"
                continue
            if "embedding" not in self.table_props(label):
                status["vector"][label] = "missing column"
                continue
            if on_progress:
                on_progress(f"vector index {label}")
            try:
                self.execute(f"CALL CREATE_VECTOR_INDEX('{label}', '{name}', 'embedding', "
                             f"metric := 'cosine')")
                status["vector"][label] = "ok"
            except Exception as exc:  # noqa: BLE001 - search falls back to brute force
                status["vector"][label] = f"error: {exc}"
        for label, (name, props) in self.FTS_INDEXES.items():
            if name in have:
                status["fts"][label] = "ok"
                continue
            tp = self.table_props(label)
            cols = [p for p in props if p in tp]
            if not cols:
                status["fts"][label] = "missing column"
                continue
            if on_progress:
                on_progress(f"fts index {label}")
            try:
                cols_sql = ", ".join(f"'{c}'" for c in cols)
                self.execute(f"CALL CREATE_FTS_INDEX('{label}', '{name}', [{cols_sql}])")
                status["fts"][label] = "ok"
            except Exception as exc:  # noqa: BLE001
                status["fts"][label] = f"error: {exc}"
        return status

    def drop_search_indexes(self) -> None:
        have = self.list_indexes()
        for label, name in self.VECTOR_INDEXES.items():
            if name in have:
                try:
                    self.execute(f"CALL DROP_VECTOR_INDEX('{label}', '{name}')")
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
            f"CALL QUERY_VECTOR_INDEX('{label}', '{name}', $v, {int(k)}{opt}) "
            f"RETURN node.id AS id, distance AS d", {"v": v})
        return [(int(r["id"]), 1.0 - float(r["d"])) for r in rows]

    def fts_topk(self, label: str, query: str, k: int) -> list[tuple[int, float]]:
        """[(id, bm25_score)] best-first via Kuzu FTS (disjunctive)."""
        name = self.FTS_INDEXES[label][0]
        if not query.strip():
            return []
        rows = self.fetch_all(
            f"CALL QUERY_FTS_INDEX('{label}', '{name}', $q, conjunctive := false, top := {int(k)}) "
            f"RETURN node.id AS id, score AS s ORDER BY s DESC LIMIT {int(k)}", {"q": query})
        return [(int(r["id"]), float(r["s"])) for r in rows]

    def fetch_arrow(self, cypher: str, params: dict | None = None, chunk: int = 1_000_000):
        """Query result as one pyarrow Table (columnar; embeddings come back
        as a fixed-size-list column that converts to numpy without Python
        float objects)."""
        result = self.execute(cypher, params)
        return result.get_as_arrow(chunk)

    def execute(self, cypher: str, params: dict | None = None) -> Any:
        if self.conn is None:
            raise DatabaseBusy(
                f"graph DB busy: connection to {self.db_path} is closed "
                "(writer active — retry shortly)"
            )
        return self.conn.execute(cypher, params or {})

    def fetch_all(self, cypher: str, params: dict | None = None) -> list[dict]:
        result = self.execute(cypher, params)
        out: list[dict] = []
        while result.has_next():
            row = result.get_next()
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

    def _ensure_known_ids(self, table: str) -> set[int]:
        """Lazy-load the set of existing primary-key ids for a node table.
        Cached on the instance; mutated by `insert_nodes` after first load."""
        ids = self._known_ids.get(table)
        if ids is None:
            ids = set()
            for r in self.fetch_all(f"MATCH (n:{table}) RETURN n.id AS id"):
                ids.add(r["id"])
            self._known_ids[table] = ids
        return ids

    def insert_edges(
        self,
        edge: str,
        from_table: str,
        to_table: str,
        rows: Iterable[dict],
        batch_size: int = 10_000,
        on_progress: "Callable[[int], None] | None" = None,
    ) -> int:
        """Bulk edge insert via Kuzu's COPY FROM (Arrow path).

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

        from_ids = self._ensure_known_ids(from_table)
        to_ids = self._ensure_known_ids(to_table)
        valid = [
            r for r in rows
            if r["from_id"] in from_ids and r["to_id"] in to_ids
        ]
        if not valid:
            return 0

        prop_keys = [k for k in valid[0].keys() if k not in ("from_id", "to_id")]
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
            if on_progress is not None:
                on_progress(len(slab))
        return n

    def close(self) -> None:
        """Explicitly close the Kuzu connection + database. Required after
        a write session so a subsequent `read_only=True` open can acquire
        the file lock — GC isn't reliable on Windows + COPY FROM holds extra
        internal references that survive a `del`."""
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
