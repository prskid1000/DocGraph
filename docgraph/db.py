"""Kuzu storage layer. One file per repo at .docgraph/graph.kuzu.

Schema covers all Tier 1-4 relationships.
"""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import kuzu

# Node tables — all entities share an integer pk for fast joins; qname is the
# stable human-readable identifier.
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
        embedding DOUBLE[384],
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
        embedding DOUBLE[384],
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
]


class GraphDB:
    def __init__(self, db_path: Path, embedding_dim: int = 384, read_only: bool = False):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.db = kuzu.Database(str(self.db_path), read_only=read_only)
        self.conn = kuzu.Connection(self.db)

    def init_schema(self) -> None:
        for ddl in NODE_DDL + EDGE_DDL:
            self.conn.execute(ddl)

    def execute(self, cypher: str, params: dict | None = None) -> Any:
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

    def insert_nodes(self, table: str, rows: Iterable[dict]) -> int:
        """Bulk insert via UNWIND. Faster than per-row CREATE."""
        rows = list(rows)
        if not rows:
            return 0
        keys = list(rows[0].keys())
        cols = ", ".join(f"{k}: row.{k}" for k in keys)
        cypher = f"UNWIND $rows AS row CREATE (n:{table} {{{cols}}})"
        self.execute(cypher, {"rows": rows})
        return len(rows)

    def insert_edges(self, edge: str, from_table: str, to_table: str, rows: Iterable[dict]) -> int:
        """Bulk edge insert. rows: [{from_id, to_id, ...edge_props}]"""
        rows = list(rows)
        if not rows:
            return 0
        extra_keys = [k for k in rows[0].keys() if k not in ("from_id", "to_id")]
        prop_clause = ""
        if extra_keys:
            prop_clause = " {" + ", ".join(f"{k}: row.{k}" for k in extra_keys) + "}"
        cypher = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{from_table} {{id: row.from_id}}), (b:{to_table} {{id: row.to_id}}) "
            f"CREATE (a)-[:{edge}{prop_clause}]->(b)"
        )
        self.execute(cypher, {"rows": rows})
        return len(rows)

    def close(self) -> None:
        # kuzu.Connection has no explicit close; rely on GC.
        pass

    @staticmethod
    def wipe(db_path: Path) -> None:
        if db_path.exists():
            if db_path.is_dir():
                shutil.rmtree(db_path)
            else:
                db_path.unlink()
