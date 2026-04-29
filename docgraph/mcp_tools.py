"""MCP tools — 6 tools, simple and tight."""
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
    retriever = Retriever(db, embedder)

    @mcp.tool()
    def search(query: str, kind: str | None = None, limit: int = 10) -> list[dict]:
        """Hybrid search for code entities by natural-language query.
        kind: 'function' | 'class' | None (both)"""
        return retriever.search(query, kind=kind, limit=limit)

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

    return mcp
