"""Hybrid retrieval: vector + name match + graph expansion + PageRank rerank.

All Cypher queries here are Kuzu-flavored:
  - `label(x)` works for both nodes and relationships
  - no `type(r)`, `startNode()`, `endNode()`, `relationships(path)`
"""
from __future__ import annotations

import numpy as np

from docgraph.db import GraphDB
from docgraph.embed import Embedder


class Retriever:
    def __init__(self, db: GraphDB, embedder: Embedder):
        self.db = db
        self.embedder = embedder

    def search(self, query: str, kind: str | None = None, limit: int = 10) -> list[dict]:
        qvec = self.embedder.embed([query])[0]
        results: list[dict] = []
        labels = ("Function",) if kind == "function" else ("Class",) if kind == "class" else ("Function", "Class")
        for label in labels:
            rows = self.db.fetch_all(
                f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, n.qname AS qname, "
                f"n.file AS file, n.line_start AS line_start, n.body AS body, "
                f"n.embedding AS embedding, n.pagerank AS pagerank"
            )
            if not rows:
                continue
            mat = np.array([r["embedding"] for r in rows], dtype=np.float32)
            qv = np.array(qvec, dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-9)
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
            sims = mat @ qv
            qlow = query.lower()
            for r, s in zip(rows, sims.tolist()):
                name_boost = 0.3 if qlow in r["name"].lower() else 0.0
                pr = r.get("pagerank") or 0.0
                score = s + name_boost + pr * 0.1
                results.append({
                    "label": label,
                    "id": r["id"],
                    "name": r["name"],
                    "qname": r["qname"],
                    "file": r["file"],
                    "line": r["line_start"],
                    "snippet": (r["body"] or "")[:300],
                    "score": float(score),
                    "pagerank": float(pr),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def definition(self, name: str, file: str | None = None) -> list[dict]:
        params: dict = {"name": name}
        where = "n.name = $name"
        if file:
            where += " AND n.file = $file"
            params["file"] = file
        rows = []
        for label in ("Function", "Class"):
            for r in self.db.fetch_all(
                f"MATCH (n:{label}) WHERE {where} "
                f"RETURN n.id AS id, n.name AS name, n.qname AS qname, "
                f"n.file AS file, n.line_start AS line, n.body AS body",
                params,
            ):
                r["label"] = label
                rows.append(r)
        return rows

    def references(self, name: str) -> list[dict]:
        out: list[dict] = []
        for edge in ("CALLS", "REFERENCES_", "INSTANTIATES"):
            try:
                rows = self.db.fetch_all(
                    f"MATCH (target)<-[r:{edge}]-(src) WHERE target.name = $name "
                    f"RETURN src.qname AS caller, src.name AS caller_name, src.file AS file, "
                    f"src.line_start AS line, label(src) AS caller_kind",
                    {"name": name},
                )
                for r in rows:
                    r["edge"] = edge
                    out.append(r)
            except Exception:
                pass
        return out

    def call_graph(self, name: str, depth: int = 2) -> dict:
        depth = max(1, min(depth, 5))
        # Forward calls
        try:
            forward_rows = self.db.fetch_all(
                f"MATCH path = (start:Function)-[:CALLS*1..{depth}]->(callee) "
                f"WHERE start.name = $name "
                f"UNWIND nodes(path) AS n RETURN DISTINCT n.qname AS qname, n.name AS name, "
                f"n.file AS file, n.line_start AS line",
                {"name": name},
            )
        except Exception:
            forward_rows = []
        # Backward callers
        try:
            backward_rows = self.db.fetch_all(
                f"MATCH path = (caller)-[:CALLS*1..{depth}]->(start:Function) "
                f"WHERE start.name = $name "
                f"UNWIND nodes(path) AS n RETURN DISTINCT n.qname AS qname, n.name AS name, "
                f"n.file AS file, n.line_start AS line",
                {"name": name},
            )
        except Exception:
            backward_rows = []
        # Direct edges (1 hop) for an actual edge list
        edges = self.db.fetch_all(
            "MATCH (a:Function)-[:CALLS]->(b:Function) "
            "WHERE a.name = $name OR b.name = $name "
            "RETURN a.qname AS src, b.qname AS dst",
            {"name": name},
        )
        return {"calls": forward_rows, "called_by": backward_rows, "edges": edges}

    def file_map(self, file: str) -> dict:
        entities: list[dict] = []
        for label in ("Function", "Class"):
            for r in self.db.fetch_all(
                f"MATCH (n:{label}) WHERE n.file = $file "
                f"RETURN n.name AS name, n.qname AS qname, n.line_start AS line, "
                f"n.pagerank AS pagerank ORDER BY n.line_start",
                {"file": file},
            ):
                r["kind"] = label
                entities.append(r)
        entities.sort(key=lambda x: x["line"])
        imports_file = self.db.fetch_all(
            "MATCH (f:File)-[:IMPORTS]->(m:File) WHERE f.path = $file "
            "RETURN m.path AS target, 'File' AS kind",
            {"file": file},
        )
        imports_mod = self.db.fetch_all(
            "MATCH (f:File)-[:IMPORTS]->(m:Module) WHERE f.path = $file "
            "RETURN m.name AS target, 'Module' AS kind",
            {"file": file},
        )
        return {"entities": entities, "imports": imports_file + imports_mod}

    def neighborhood(self, name: str, limit: int = 10) -> list[dict]:
        out: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for edge in ("CALLS", "REFERENCES_", "SIMILAR_TO", "INHERITS", "TESTS"):
            try:
                rows = self.db.fetch_all(
                    f"MATCH (n)-[r:{edge}]-(other) WHERE n.name = $name AND other.name IS NOT NULL "
                    f"RETURN DISTINCT other.qname AS qname, other.name AS name, other.file AS file, "
                    f"other.line_start AS line, label(other) AS kind, "
                    f"coalesce(other.pagerank, 0.0) AS pagerank",
                    {"name": name},
                )
                for r in rows:
                    key = (r["qname"], 0)
                    if key in seen:
                        continue
                    seen.add(key)
                    r["via"] = edge
                    out.append(r)
            except Exception:
                pass
        out.sort(key=lambda x: x.get("pagerank") or 0.0, reverse=True)
        return out[:limit]

    def graph_dump(self, limit_nodes: int = 2000) -> dict:
        nodes: list[dict] = []
        # Functions and classes
        for label in ("Function", "Class"):
            rows = self.db.fetch_all(
                f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name, n.file AS file, "
                f"coalesce(n.pagerank, 0.0) AS pagerank ORDER BY pagerank DESC LIMIT {limit_nodes}",
            )
            for r in rows:
                r["kind"] = label
                nodes.append(r)
        # Files (use path as name)
        rows = self.db.fetch_all(
            f"MATCH (n:File) RETURN n.id AS id, n.path AS name, n.path AS file, "
            f"coalesce(n.pagerank, 0.0) AS pagerank ORDER BY pagerank DESC LIMIT {limit_nodes}",
        )
        for r in rows:
            r["kind"] = "File"
            nodes.append(r)
        # Trim to top N by pagerank
        nodes.sort(key=lambda x: x.get("pagerank") or 0.0, reverse=True)
        nodes = nodes[:limit_nodes]
        node_ids = {n["id"] for n in nodes}

        edges: list[dict] = []
        for edge in ("CALLS", "INHERITS", "REFERENCES_", "IMPORTS",
                     "SIMILAR_TO", "TESTS", "CO_CHANGED_WITH", "INSTANTIATES"):
            try:
                rows = self.db.fetch_all(
                    f"MATCH (a)-[r:{edge}]->(b) RETURN a.id AS src, b.id AS dst"
                )
                for r in rows:
                    if r["src"] in node_ids and r["dst"] in node_ids:
                        r["kind"] = edge
                        edges.append(r)
            except Exception:
                pass
        return {"nodes": nodes, "edges": edges}
