"""MCP tools — multi-root.

15 base tools + `list_roots`. Every tool that hits the graph takes a
`root` argument typed as a dynamic enum built from the slugs the host was
started with, so the LLM picks from a closed set and the schema rejects
typos at the protocol layer.

Single-root case: the enum has one value and is also the default — the
LLM doesn't need to think about it; calls with `root` omitted just work.

Note: this module deliberately does NOT use `from __future__ import
annotations`. Each tool's `root` parameter is annotated with the
dynamically-built enum class, which is a local in `make_mcp`. With
deferred annotation evaluation, Pydantic's later `get_type_hints` can't
see the local class. Eager evaluation captures the closure correctly.
"""
import enum
from typing import Any, Optional, Union

from fastmcp import FastMCP

from docgraph.workspace import Workspace


def _root_enum(workspace: Workspace) -> type[enum.Enum]:
    """Build an `Enum` whose values are the workspace's slugs.

    We subclass `(str, Enum)` instead of `StrEnum` because `StrEnum` is
    Python 3.11+ and our floor is 3.10. Pydantic emits a JSON Schema
    with `type: string, enum: [...]` for `(str, Enum)` subclasses, which
    is what FastMCP forwards to the client.
    """
    members = {s.upper().replace("-", "_"): s for s in workspace.slugs()}
    if not members:
        raise ValueError("Workspace has no roots; cannot build MCP tool surface")
    return enum.Enum("RootSlug", members, type=str)  # type: ignore[arg-type]


def make_mcp(workspace: Workspace) -> FastMCP:
    mcp: FastMCP = FastMCP(name="docgraph")
    RootSlug = _root_enum(workspace)
    DEFAULT = RootSlug(workspace.default_slug())

    def _retriever(root):
        # Members of `(str, Enum)` subclasses have a `.value` attribute
        # holding the actual slug. `str(member)` would give 'RootSlug.X'
        # instead, which is what we don't want.
        slug = root.value if hasattr(root, "value") else str(root)
        return workspace.resolve(slug).retriever

    @mcp.tool()
    def list_roots() -> list[dict]:
        """List every registered root: slug, absolute path, default flag,
        watching flag, last_indexed_at timestamp. Use this to discover
        what the host was started with."""
        return workspace.list()

    @mcp.tool()
    def search(
        query: str,
        kind: str | None = None,
        limit: int = 10,
        focus_file: str | None = None,
        focus_symbol: str | None = None,
        rerank: bool = False,
        root: RootSlug = DEFAULT,
    ) -> list[dict]:
        """Hybrid search for code entities by natural-language query.
        kind: 'function' | 'class' | None (both).
        focus_file / focus_symbol: bias ranking toward the agent's current
        location via personalized PageRank.
        rerank: run a cross-encoder over the top candidates for higher
        precision (downloads a ~33 MB model on first use).
        root: which registered root to query."""
        return _retriever(root).search(
            query, kind=kind, limit=limit,
            focus_file=focus_file, focus_symbol=focus_symbol,
            rerank=rerank,
        )

    @mcp.tool()
    def definition(name: str, file: str | None = None,
                   root: RootSlug = DEFAULT) -> list[dict]:
        """Get definition + body of a symbol. Optionally scoped to a file."""
        return _retriever(root).definition(name, file=file)

    @mcp.tool()
    def references(name: str, root: RootSlug = DEFAULT) -> list[dict]:
        """All callers and other references of a symbol."""
        return _retriever(root).references(name)

    @mcp.tool()
    def call_graph(name: str, depth: int = 2, root: RootSlug = DEFAULT) -> dict:
        """Forward + backward call graph for a function. depth in [1,5]."""
        return _retriever(root).call_graph(name, depth=depth)

    @mcp.tool()
    def file_map(file: str, root: RootSlug = DEFAULT) -> dict:
        """All entities and imports in a file, ordered by line."""
        return _retriever(root).file_map(file)

    @mcp.tool()
    def neighborhood(name: str, limit: int = 10, root: RootSlug = DEFAULT) -> list[dict]:
        """Related entities via call graph, similarity, inheritance, and tests.
        PageRank-ordered. The 'what else should I read?' tool."""
        return _retriever(root).neighborhood(name, limit=limit)

    @mcp.tool()
    def explore(seeds: list[str], hops: int = 3, limit: int = 25,
                root: RootSlug = DEFAULT) -> dict:
        """Multi-hop graph walk from seed symbols. Returns the relevant
        subgraph in one call so the agent doesn't need to chain neighborhood
        lookups. seeds: list of Function/Class names. hops in [1,5]."""
        return _retriever(root).explore(seeds=seeds, hops=hops, limit=limit)

    @mcp.tool()
    def impact_of(target: str, depth: int = 3, limit: int = 50,
                  root: RootSlug = DEFAULT) -> dict:
        """Blast radius of a file or symbol: transitive callers, importers,
        co-changed files, and tests. Use before refactoring or to scope a PR
        review."""
        return _retriever(root).impact_of(target, depth=depth, limit=limit)

    @mcp.tool()
    def test_impact(target: str, limit: int = 25, root: RootSlug = DEFAULT) -> list[dict]:
        """Tests that exercise the given file or symbol. Combines explicit
        TESTS edges with transitive CALLS reverse traversal from test
        functions."""
        return _retriever(root).test_impact(target, limit=limit)

    @mcp.tool()
    def git_changes(ref: str | None = None, root: RootSlug = DEFAULT) -> dict:
        """Files + entities touched by a git diff, plus 1-hop callers of the
        changed functions. ref: None (working tree), 'HEAD' (last commit),
        'main' (branch vs main), or a commit SHA."""
        return _retriever(root).git_changes(ref=ref)

    @mcp.tool()
    def git_blame(file: str, line_start: int = 1, line_end: int | None = None,
                  root: RootSlug = DEFAULT) -> list[dict]:
        """`git blame` for a file or line range. Returns commit + author +
        date per line. Mirrors Cursor Blame."""
        return _retriever(root).git_blame(file, line_start=line_start, line_end=line_end)

    @mcp.tool()
    def git_recent(file: str | None = None, limit: int = 20,
                   root: RootSlug = DEFAULT) -> list[dict]:
        """Recent commits across the repo or scoped to one file."""
        return _retriever(root).git_recent(file=file, limit=limit)

    @mcp.tool()
    def search_docs(query: str, limit: int = 10, root: RootSlug = DEFAULT) -> list[dict]:
        """Semantic search across ingested external documentation.
        Add docs first: `docgraph docs add <url>`. Cursor @Docs parity."""
        return _retriever(root).search_docs(query, limit=limit)

    @mcp.tool()
    def rules_for(file: str, root: RootSlug = DEFAULT) -> list[dict]:
        """Auto-attach rules for `file`: matches .cursor/rules/*.mdc by glob,
        plus AGENTS.md / CLAUDE.md as always-on. Compatible with the Cursor
        Rules ecosystem — drop in existing .mdc files and they work here."""
        return _retriever(root).rules_for(file)

    @mcp.tool()
    def cypher(query: str, limit: int = 100, root: RootSlug = DEFAULT) -> dict:
        """Run a READ-ONLY Cypher query against the graph. Rejects writes.
        Schema: nodes (File, Module, Class, Function, Variable);
        edges (CONTAINS, IMPORTS, CALLS, INSTANTIATES, REFERENCES_, INHERITS,
        DECORATED_BY, SIMILAR_TO, CO_CHANGED_WITH, TESTS).
        Use n.id / n.name / n.file. File node uses .path not .name."""
        return _retriever(root).cypher(query, limit=limit)

    return mcp
