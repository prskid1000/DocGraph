"""MCP tools.

The original 6 tools (search, definition, references, call_graph, file_map,
neighborhood) plus 9 differentiators that no other indexer exposes:
- explore: multi-hop graph walk from one or more seeds
- impact_of: blast-radius of a file/symbol (callers + importers + co_changed + tests)
- test_impact: which tests exercise this code (TESTS + reverse CALLS*)
- cypher: read-only Cypher escape hatch for power agents
- git_changes / git_blame / git_recent: diff-aware retrieval + Cursor blame parity
- rules_for: glob-matches `.cursor/rules/*.mdc` + AGENTS.md / CLAUDE.md
- search_docs: semantic search over `docgraph docs add <url>`-ingested external docs
"""
from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.retrieve import Retriever


def make_mcp(cfg: Config) -> FastMCP:
    mcp: FastMCP = FastMCP(name="docgraph")
    db = GraphDB(cfg.db_path, read_only=True)
    embedder = Embedder(cfg.embedding_model)
    retriever = Retriever(db, embedder, cfg=cfg)

    @mcp.tool()
    def search(
        query: str,
        kind: str | None = None,
        limit: int = 10,
        focus_file: str | None = None,
        focus_symbol: str | None = None,
        rerank: bool = False,
    ) -> list[dict]:
        """Hybrid search for code entities by natural-language query.
        kind: 'function' | 'class' | None (both).
        focus_file / focus_symbol: bias ranking toward the agent's current
        location via personalized PageRank.
        rerank: run a cross-encoder over the top candidates for higher
        precision (downloads a ~33 MB model on first use)."""
        return retriever.search(
            query, kind=kind, limit=limit,
            focus_file=focus_file, focus_symbol=focus_symbol,
            rerank=rerank,
        )

    @mcp.tool()
    def definition(name: str, file: str | None = None) -> list[dict]:
        """Get definition + body of a symbol. Optionally scoped to a file."""
        return retriever.definition(name, file=file)

    @mcp.tool()
    def references(name: str) -> list[dict]:
        """All callers and other references of a symbol."""
        return retriever.references(name)

    @mcp.tool()
    def call_graph(name: str, depth: int = 2) -> dict:
        """Forward + backward call graph for a function. depth in [1,5]."""
        return retriever.call_graph(name, depth=depth)

    @mcp.tool()
    def file_map(file: str) -> dict:
        """All entities and imports in a file, ordered by line."""
        return retriever.file_map(file)

    @mcp.tool()
    def neighborhood(name: str, limit: int = 10) -> list[dict]:
        """Related entities via call graph, similarity, inheritance, and tests.
        PageRank-ordered. The 'what else should I read?' tool."""
        return retriever.neighborhood(name, limit=limit)

    @mcp.tool()
    def explore(
        seeds: list[str],
        hops: int = 3,
        limit: int = 25,
    ) -> dict:
        """Multi-hop graph walk from seed symbols. Returns the relevant
        subgraph in one call so the agent doesn't need to chain neighborhood
        lookups. seeds: list of Function/Class names. hops in [1,5]."""
        return retriever.explore(seeds=seeds, hops=hops, limit=limit)

    @mcp.tool()
    def impact_of(target: str, depth: int = 3, limit: int = 50) -> dict:
        """Blast radius of a file or symbol: transitive callers, importers,
        co-changed files, and tests. Use before refactoring or to scope a PR
        review."""
        return retriever.impact_of(target, depth=depth, limit=limit)

    @mcp.tool()
    def test_impact(target: str, limit: int = 25) -> list[dict]:
        """Tests that exercise the given file or symbol. Combines explicit
        TESTS edges with transitive CALLS reverse traversal from test
        functions."""
        return retriever.test_impact(target, limit=limit)

    @mcp.tool()
    def git_changes(ref: str | None = None) -> dict:
        """Files + entities touched by a git diff, plus 1-hop callers of the
        changed functions. ref: None (working tree), 'HEAD' (last commit),
        'main' (branch vs main), or a commit SHA. Use this to scope a code
        review, debug 'what did I break?', or surface PR context."""
        return retriever.git_changes(ref=ref)

    @mcp.tool()
    def git_blame(file: str, line_start: int = 1, line_end: int | None = None) -> list[dict]:
        """`git blame` for a file or line range. Returns commit + author +
        date per line. Mirrors Cursor Blame."""
        return retriever.git_blame(file, line_start=line_start, line_end=line_end)

    @mcp.tool()
    def git_recent(file: str | None = None, limit: int = 20) -> list[dict]:
        """Recent commits across the repo or scoped to one file."""
        return retriever.git_recent(file=file, limit=limit)

    @mcp.tool()
    def search_docs(query: str, limit: int = 10) -> list[dict]:
        """Semantic search across ingested external documentation.
        Add docs first: `docgraph docs add <url>`. Cursor @Docs parity."""
        return retriever.search_docs(query, limit=limit)

    @mcp.tool()
    def rules_for(file: str) -> list[dict]:
        """Auto-attach rules for `file`: matches .cursor/rules/*.mdc by glob,
        plus AGENTS.md / CLAUDE.md as always-on. Compatible with the Cursor
        Rules ecosystem — drop in existing .mdc files and they work here."""
        return retriever.rules_for(file)

    @mcp.tool()
    def cypher(query: str, limit: int = 100) -> dict:
        """Run a READ-ONLY Cypher query against the graph. Rejects writes.
        Schema: nodes (File, Module, Class, Function, Variable);
        edges (CONTAINS, IMPORTS, CALLS, INSTANTIATES, REFERENCES_, INHERITS,
        DECORATED_BY, SIMILAR_TO, CO_CHANGED_WITH, TESTS).
        Use n.id / n.name / n.file. File node uses .path not .name."""
        return retriever.cypher(query, limit=limit)

    return mcp
