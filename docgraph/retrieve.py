"""Hybrid retrieval: vector + name match + graph expansion + PageRank rerank.

All Cypher queries here are Kuzu-flavored:
  - `label(x)` works for both nodes and relationships
  - no `type(r)`, `startNode()`, `endNode()`, `relationships(path)`
"""
from __future__ import annotations

import re

import numpy as np

from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.git_tools import blame_lines, changed_entities, recent_commits
from docgraph.rank import PersonalizedRanker
from docgraph.rerank import Reranker
from docgraph.rules import rules_for as _rules_for


class Retriever:
    def __init__(self, db: GraphDB, embedder: Embedder, cfg: Config | None = None):
        self.db = db
        self.embedder = embedder
        self.cfg = cfg
        self._ranker: PersonalizedRanker | None = None
        self._reranker: Reranker | None = None

    def _reranker_(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def _ranker_(self) -> PersonalizedRanker:
        if self._ranker is None:
            self._ranker = PersonalizedRanker(self.db)
        return self._ranker

    def _chunk_max_sims(self, qvec) -> dict[str, float]:
        """For each parent_qname, the best cosine similarity across its
        sub-chunks. Empty when no chunks exist."""
        try:
            rows = self.db.fetch_all(
                "MATCH (c:Chunk) RETURN c.parent_qname AS qname, c.embedding AS embedding"
            )
        except Exception:
            return {}
        if not rows:
            return {}
        mat = np.array([r["embedding"] for r in rows], dtype=np.float32)
        qv = np.array(qvec, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat = mat / norms
        sims = (mat @ qv).tolist()
        out: dict[str, float] = {}
        for r, s in zip(rows, sims):
            q = r["qname"]
            if q not in out or s > out[q]:
                out[q] = float(s)
        return out

    def _redact(self, file: str | None, body: str | None, snippet: str | None = None) -> tuple[str | None, str | None]:
        """Mask body/snippet if the file is AI-blocked. Returns (body, snippet)."""
        if not file or self.cfg is None:
            return body, snippet
        if self.cfg.ai_blocked_logical(file):
            return "[redacted by .cursorignore]", "[redacted]"
        return body, snippet

    def search(
        self,
        query: str,
        kind: str | None = None,
        limit: int = 10,
        focus_file: str | None = None,
        focus_symbol: str | None = None,
        rerank: bool = False,
    ) -> list[dict]:
        """Hybrid search. If focus_file or focus_symbol is provided, ranks
        results by personalized PageRank biased toward that focus point —
        the model sees results most relevant to where the agent is working.

        rerank=True runs a cross-encoder over the top candidates for
        token-level precision (downloads a small ~33 MB model on first use).
        """
        qvec = self.embedder.embed([query])[0]
        results: list[dict] = []
        labels = ("Function",) if kind == "function" else ("Class",) if kind == "class" else ("Function", "Class")

        ppr = self._maybe_ppr(focus_file, focus_symbol)

        # Per-entity max chunk similarity (sub-function chunking lift):
        # for any qname, the best score across its sub-chunks rivals the
        # entity-level score so a query that matches a small piece of a
        # 500-line function still surfaces it.
        chunk_max = self._chunk_max_sims(self.embedder.embed([query])[0])

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
                ppr_boost = ppr.get(r["id"], 0.0) if ppr else 0.0
                # Use personalized PR when present; fall back to global.
                rank_term = (ppr_boost * 0.5) if ppr else (pr * 0.1)
                # Take max(entity_sim, best_chunk_sim) so long-body entities
                # don't lose recall when only one section matches the query.
                best_sim = max(s, chunk_max.get(r["qname"], -1.0))
                score = best_sim + name_boost + rank_term
                s = best_sim  # for the returned `score` field below
                _, snippet = self._redact(r["file"], None, (r["body"] or "")[:300])
                results.append({
                    "label": label,
                    "id": r["id"],
                    "name": r["name"],
                    "qname": r["qname"],
                    "file": r["file"],
                    "line": r["line_start"],
                    "snippet": snippet,
                    "score": float(score),
                    "pagerank": float(pr),
                    "ppr": float(ppr_boost),
                })
        results.sort(key=lambda x: x["score"], reverse=True)

        if rerank and results:
            try:
                results = self._reranker_().rerank(
                    query, results, text_key="snippet", top_k=50,
                )
            except Exception as e:  # noqa: BLE001
                # Don't fail the search if the reranker can't load (offline,
                # no model, etc.) — degrade silently to bi-encoder ranking.
                import logging
                logging.getLogger(__name__).warning(f"Rerank failed, falling back: {e}")
        return results[:limit]

    def _focus_ids(self, focus_file: str | None, focus_symbol: str | None) -> list[int]:
        """Translate a file path or symbol name into seed node IDs."""
        ids: list[int] = []
        if focus_file:
            for label, prop in (("File", "path"), ("Function", "file"), ("Class", "file")):
                for r in self.db.fetch_all(
                    f"MATCH (n:{label}) WHERE n.{prop} = $f RETURN n.id AS id",
                    {"f": focus_file},
                ):
                    ids.append(r["id"])
        if focus_symbol:
            for label in ("Function", "Class"):
                for r in self.db.fetch_all(
                    f"MATCH (n:{label}) WHERE n.name = $s RETURN n.id AS id",
                    {"s": focus_symbol},
                ):
                    ids.append(r["id"])
        return ids

    def _maybe_ppr(
        self, focus_file: str | None, focus_symbol: str | None
    ) -> dict[int, float] | None:
        if not focus_file and not focus_symbol:
            return None
        ids = self._focus_ids(focus_file, focus_symbol)
        if not ids:
            return None
        try:
            return self._ranker_().personalized(ids)
        except Exception:
            return None

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
                body, _ = self._redact(r.get("file"), r.get("body"))
                r["body"] = body
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

    # --- Multi-hop / impact / test_impact / cypher ---------------------

    def explore(
        self,
        seeds: list[str],
        hops: int = 3,
        limit: int = 25,
        edges: tuple[str, ...] = ("CALLS", "REFERENCES_", "SIMILAR_TO", "INHERITS", "TESTS"),
    ) -> dict:
        """Multi-hop graph walk from one or more seed names. Returns nodes
        ranked by min-distance and pagerank — the agent gets a 1-shot view of
        the relevant subgraph instead of having to chain `neighborhood` calls.

        seeds: symbol names (Function/Class). hops: 1..5.
        """
        hops = max(1, min(int(hops), 5))
        if not seeds:
            return {"nodes": [], "edges": []}

        # Resolve seed names to IDs (Function or Class)
        seed_ids: list[int] = []
        for s in seeds:
            for r in self.db.fetch_all(
                "MATCH (n) WHERE (label(n) = 'Function' OR label(n) = 'Class') AND n.name = $s "
                "RETURN n.id AS id",
                {"s": s},
            ):
                seed_ids.append(r["id"])
        if not seed_ids:
            return {"nodes": [], "edges": []}

        # BFS — each level we expand via every requested edge type.
        seen: dict[int, int] = {sid: 0 for sid in seed_ids}  # id → min-distance
        frontier = set(seed_ids)
        edge_records: list[dict] = []
        for d in range(1, hops + 1):
            if not frontier:
                break
            next_frontier: set[int] = set()
            for edge in edges:
                try:
                    rows = self.db.fetch_all(
                        f"MATCH (a)-[r:{edge}]-(b) WHERE a.id IN $ids "
                        f"RETURN a.id AS src, b.id AS dst",
                        {"ids": list(frontier)},
                    )
                except Exception:
                    continue
                for row in rows:
                    edge_records.append({"src": row["src"], "dst": row["dst"], "kind": edge})
                    if row["dst"] not in seen:
                        seen[row["dst"]] = d
                        next_frontier.add(row["dst"])
            frontier = next_frontier

        if not seen:
            return {"nodes": [], "edges": []}

        rows = self.db.fetch_all(
            "MATCH (n) WHERE n.id IN $ids AND (label(n) = 'Function' OR label(n) = 'Class') "
            "RETURN n.id AS id, n.name AS name, n.qname AS qname, n.file AS file, "
            "n.line_start AS line, label(n) AS kind, "
            "coalesce(n.pagerank, 0.0) AS pagerank",
            {"ids": list(seen.keys())},
        )
        nodes = []
        for r in rows:
            d = seen[r["id"]]
            # Higher score = closer + more central
            score = (1.0 / (d + 1)) + (r["pagerank"] or 0.0) * 0.5
            r["distance"] = d
            r["score"] = score
            nodes.append(r)
        nodes.sort(key=lambda x: x["score"], reverse=True)
        return {"nodes": nodes[:limit], "edges": edge_records}

    def impact_of(
        self,
        target: str,
        depth: int = 3,
        limit: int = 50,
    ) -> dict:
        """Blast radius of a file or symbol. Returns:
          - callers: transitive callers (CALLS reverse, up to `depth` hops)
          - importers: files that import this file
          - co_changed: files that historically changed alongside
          - tests: tests that exercise the target

        target: a symbol name OR a file path. We try file first, then symbol.
        """
        depth = max(1, min(int(depth), 5))
        out: dict = {"target": target, "callers": [], "importers": [], "co_changed": [], "tests": []}

        is_file = bool(self.db.fetch_all(
            "MATCH (f:File) WHERE f.path = $t RETURN f.id LIMIT 1", {"t": target}
        ))

        if is_file:
            # Importers
            out["importers"] = self.db.fetch_all(
                "MATCH (a:File)-[:IMPORTS]->(b:File) WHERE b.path = $t "
                "RETURN a.path AS file",
                {"t": target},
            )
            # Co-changed
            out["co_changed"] = self.db.fetch_all(
                "MATCH (a:File)-[r:CO_CHANGED_WITH]-(b:File) WHERE a.path = $t "
                "RETURN b.path AS file, r.count AS count ORDER BY r.count DESC LIMIT 25",
                {"t": target},
            )
            # Transitive callers of any function in this file
            try:
                rows = self.db.fetch_all(
                    f"MATCH path = (caller:Function)-[:CALLS*1..{depth}]->(callee:Function) "
                    f"WHERE callee.file = $t "
                    f"UNWIND nodes(path) AS n "
                    f"RETURN DISTINCT n.qname AS qname, n.name AS name, n.file AS file, "
                    f"n.line_start AS line, coalesce(n.pagerank,0.0) AS pagerank "
                    f"ORDER BY pagerank DESC LIMIT $lim",
                    {"t": target, "lim": limit},
                )
                out["callers"] = rows
            except Exception:
                out["callers"] = []
            # Tests
            try:
                out["tests"] = self.db.fetch_all(
                    "MATCH (t:Function)-[:TESTS]->(target) WHERE target.file = $t "
                    "RETURN t.name AS name, t.file AS file, t.line_start AS line "
                    "LIMIT $lim",
                    {"t": target, "lim": limit},
                )
            except Exception:
                out["tests"] = []
        else:
            # Symbol path
            try:
                out["callers"] = self.db.fetch_all(
                    f"MATCH path = (caller)-[:CALLS*1..{depth}]->(target:Function) "
                    f"WHERE target.name = $t "
                    f"UNWIND nodes(path) AS n RETURN DISTINCT n.qname AS qname, n.name AS name, "
                    f"n.file AS file, n.line_start AS line, "
                    f"coalesce(n.pagerank,0.0) AS pagerank "
                    f"ORDER BY pagerank DESC LIMIT $lim",
                    {"t": target, "lim": limit},
                )
            except Exception:
                out["callers"] = []
            try:
                out["tests"] = self.db.fetch_all(
                    "MATCH (test:Function)-[:TESTS]->(target) WHERE target.name = $t "
                    "RETURN test.name AS name, test.file AS file, test.line_start AS line "
                    "LIMIT $lim",
                    {"t": target, "lim": limit},
                )
            except Exception:
                out["tests"] = []
            # File of the symbol → its importers + co-changed
            files_of_symbol = self.db.fetch_all(
                "MATCH (n) WHERE (label(n) = 'Function' OR label(n) = 'Class') AND n.name = $t "
                "RETURN DISTINCT n.file AS file LIMIT 5",
                {"t": target},
            )
            for fr in files_of_symbol:
                f = fr["file"]
                out["importers"].extend(self.db.fetch_all(
                    "MATCH (a:File)-[:IMPORTS]->(b:File) WHERE b.path = $f RETURN a.path AS file",
                    {"f": f},
                ))
                out["co_changed"].extend(self.db.fetch_all(
                    "MATCH (a:File)-[r:CO_CHANGED_WITH]-(b:File) WHERE a.path = $f "
                    "RETURN b.path AS file, r.count AS count ORDER BY r.count DESC LIMIT 10",
                    {"f": f},
                ))
        return out

    def test_impact(self, target: str, limit: int = 25) -> list[dict]:
        """Tests that exercise `target` (file or symbol). Differentiator:
        we already have TESTS edges + reverse CALLS, no competitor exposes
        this as a primitive."""
        is_file = bool(self.db.fetch_all(
            "MATCH (f:File) WHERE f.path = $t RETURN f.id LIMIT 1", {"t": target}
        ))
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []

        def add(rows: list[dict], via: str) -> None:
            for r in rows:
                key = (r.get("name"), r.get("file"))
                if key in seen:
                    continue
                seen.add(key)
                r["via"] = via
                out.append(r)
                if len(out) >= limit:
                    return

        if is_file:
            try:
                add(self.db.fetch_all(
                    "MATCH (t:Function)-[:TESTS]->(target) WHERE target.file = $t "
                    "RETURN t.name AS name, t.file AS file, t.line_start AS line",
                    {"t": target},
                ), "TESTS")
            except Exception:
                pass
            try:
                add(self.db.fetch_all(
                    "MATCH (t:Function)-[:CALLS*1..3]->(callee:Function) "
                    "WHERE callee.file = $t AND t.is_test = true "
                    "RETURN DISTINCT t.name AS name, t.file AS file, t.line_start AS line "
                    "LIMIT $lim",
                    {"t": target, "lim": limit},
                ), "CALLS*")
            except Exception:
                pass
        else:
            try:
                add(self.db.fetch_all(
                    "MATCH (t:Function)-[:TESTS]->(target) WHERE target.name = $t "
                    "RETURN t.name AS name, t.file AS file, t.line_start AS line",
                    {"t": target},
                ), "TESTS")
            except Exception:
                pass
            try:
                add(self.db.fetch_all(
                    "MATCH (t:Function)-[:CALLS*1..3]->(callee:Function) "
                    "WHERE callee.name = $t AND t.is_test = true "
                    "RETURN DISTINCT t.name AS name, t.file AS file, t.line_start AS line "
                    "LIMIT $lim",
                    {"t": target, "lim": limit},
                ), "CALLS*")
            except Exception:
                pass
        return out[:limit]

    # Cypher escape hatch ---------------------------------------------------

    _WRITE_KEYWORDS = (
        "CREATE", "MERGE", "DELETE", "DETACH",
        "SET", "REMOVE", "DROP", "ALTER", "COPY",
    )

    @classmethod
    def _is_read_only(cls, query: str) -> bool:
        upper = query.upper()
        # Strip string literals so e.g. "MERGE" inside a string doesn't false-positive
        stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "", upper)
        for kw in cls._WRITE_KEYWORDS:
            if re.search(rf"\b{kw}\b", stripped):
                return False
        return True

    def cypher(self, query: str, limit: int = 100) -> dict:
        """Read-only Cypher escape hatch. Lets the agent author its own graph
        queries — none of the competitors expose this. Rejects writes; caps
        rows.

        Returns {"rows": [...], "rejected": str | None}.
        """
        if not self._is_read_only(query):
            return {"rows": [], "rejected": "write keyword detected (CREATE/MERGE/SET/DELETE/...)"}
        # Append a LIMIT safety net unless one exists
        if "LIMIT" not in query.upper():
            query = f"{query.rstrip(';')} LIMIT {int(limit)}"
        try:
            return {"rows": self.db.fetch_all(query)[:limit], "rejected": None}
        except Exception as e:  # noqa: BLE001
            return {"rows": [], "rejected": f"query error: {e}"}

    # --- Git-aware retrieval ----------------------------------------------

    def git_changes(self, ref: str | None = None) -> dict:
        """Diff-aware retrieval. ref:
          - None    → unstaged + staged working-tree diff
          - "HEAD"  → last commit
          - "main"  → branch diff vs main
          - "<sha>" → that commit

        Returns changed files + entities + the 1-hop callers of changed
        functions, so the agent gets a 'what's about to break' picture in one
        call. Mirrors Cursor's @Commit / @Recent Changes / @PR but joined to
        the graph.
        """
        if self.cfg is None:
            return {"ref": ref, "files": [], "entities": [], "callers_of_changed": [],
                    "error": "Config not attached to Retriever"}
        return changed_entities(self.cfg, self.db, ref)

    def git_blame(self, file: str, line_start: int = 1, line_end: int | None = None) -> list[dict]:
        """`git blame` for a file/line range. Mirrors Cursor Blame."""
        if self.cfg is None:
            return []
        return blame_lines(self.cfg, file, line_start=line_start, line_end=line_end)

    def git_recent(self, file: str | None = None, limit: int = 20) -> list[dict]:
        """Recent commits, optionally scoped to a file path."""
        if self.cfg is None:
            return []
        return recent_commits(self.cfg, file_path=file, limit=limit)

    # --- Auto-attach rules (.cursor/rules/*.mdc + AGENTS.md / CLAUDE.md) --

    def rules_for(self, file: str) -> list[dict]:
        """Cursor-rules-compatible auto-attach: return rules whose globs
        match `file`, plus AGENTS.md / CLAUDE.md as always-apply."""
        if self.cfg is None:
            return []
        return _rules_for(self.cfg, file)

    # --- @Docs (external knowledge) ---------------------------------------

    def search_docs(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search across ingested external documentation
        (`docgraph docs add <url>`). Cursor `@Docs` parity."""
        try:
            rows = self.db.fetch_all(
                "MATCH (d:Doc) RETURN d.id AS id, d.source AS source, "
                "d.title AS title, d.idx AS idx, d.body AS body, "
                "d.embedding AS embedding"
            )
        except Exception:
            return []
        if not rows:
            return []
        qvec = self.embedder.embed([query])[0]
        mat = np.array([r["embedding"] for r in rows], dtype=np.float32)
        qv = np.array(qvec, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = (mat @ qv).tolist()
        out = []
        for r, s in zip(rows, sims):
            out.append({
                "source": r["source"],
                "title": r["title"],
                "idx": r["idx"],
                "snippet": (r["body"] or "")[:600],
                "score": float(s),
            })
        out.sort(key=lambda x: x["score"], reverse=True)
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
