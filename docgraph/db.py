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
# Embedding columns are `DOUBLE[{dim}]` and substituted at init time from
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
        embedding DOUBLE[{dim}],
        pagerank DOUBLE,
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
        embedding DOUBLE[{dim}],
        pagerank DOUBLE,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Variable(
        id INT64,
        name STRING,
        qname STRING,
        file STRING,
        line INT64,
        scope STRING,
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
        embedding DOUBLE[{dim}],
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
    "CREATE REL TABLE IF NOT EXISTS CALLS(FROM Function TO Function, line INT64)",
    "CREATE REL TABLE IF NOT EXISTS INSTANTIATES(FROM Function TO Class, line INT64)",
    "CREATE REL TABLE IF NOT EXISTS REFERENCES_(FROM Function TO Class, FROM Function TO Variable, FROM Function TO Function, line INT64)",
    "CREATE REL TABLE IF NOT EXISTS RETURNS(FROM Function TO Class)",
    # Tier 3 — Type system
    "CREATE REL TABLE IF NOT EXISTS INHERITS(FROM Class TO Class)",
    "CREATE REL TABLE IF NOT EXISTS IMPLEMENTS(FROM Class TO Class)",
    "CREATE REL TABLE IF NOT EXISTS OVERRIDES(FROM Function TO Function)",
    "CREATE REL TABLE IF NOT EXISTS DECORATED_BY(FROM Function TO Function, FROM Class TO Function)",
    # Tier 4 — Differentiators
    "CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(FROM Function TO Function, FROM Class TO Class, score DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS CO_CHANGED_WITH(FROM File TO File, count INT64)",
    "CREATE REL TABLE IF NOT EXISTS TESTS(FROM Function TO Function, FROM Function TO Class)",
    "CREATE REL TABLE IF NOT EXISTS CONTAINS_CHUNK(FROM Function TO Chunk, FROM Class TO Chunk)",
]


class DatabaseBusy(RuntimeError):
    """Raised when a query hits a connection that's been closed because a
    writer (watcher reindex / index / wiki) currently owns the
    file's exclusive Kuzu lock. Routes catch this and return 503 +
    Retry-After so the client can poll until the writer releases.

    Kuzu enforces a per-DB-file write lock — we cannot keep an RO handle
    open while a writer is active in another connection — so the only
    sane behavior during a writer-held window is to refuse reads with a
    well-typed error, not crash with AttributeError on a None conn."""


class GraphDB:
    def __init__(self, db_path: Path, embedding_dim: int = 384, read_only: bool = False):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.db = kuzu.Database(str(self.db_path), read_only=read_only)
        self.conn = kuzu.Connection(self.db)
        # Per-node-table id sets, populated lazily on first edge insert. Used
        # to filter dangling-endpoint rows before COPY FROM (which errors hard
        # on unknown PKs, vs. the old MATCH+CREATE which silently dropped).
        # `insert_nodes` extends the cache so freshly inserted nodes are seen.
        self._known_ids: dict[str, set[int]] = {}

    def init_schema(self) -> None:
        # NODE_DDL templates contain `{dim}` placeholders for embedding columns
        # so the on-disk schema matches whatever model the user picked.
        for ddl in NODE_DDL:
            self.conn.execute(ddl.format(dim=self.embedding_dim))
        for ddl in EDGE_DDL:
            self.conn.execute(ddl)

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
