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

    def _slot(root):
        slug = root.value if hasattr(root, "value") else str(root)
        return workspace.resolve(slug)

    def _retriever(root):
        # Members of `(str, Enum)` subclasses have a `.value` attribute
        # holding the actual slug. `str(member)` would give 'RootSlug.X'
        # instead, which is what we don't want.
        return _slot(root).retriever

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
        rerank: bool | None = None,
        root: RootSlug = DEFAULT,
    ) -> list[dict]:
        """Hybrid search for code entities by natural-language query.
        kind: 'function' | 'class' | None (both).
        focus_file / focus_symbol: bias ranking toward the agent's current
        location via personalized PageRank.
        rerank: run a cross-encoder over the top candidates for higher
        precision. None = follow cfg.rerank_default (DOCGRAPH_RERANK_DEFAULT
        env var or telecode tray toggle); explicit True/False overrides.
        root: which registered root to query."""
        slot = _slot(root)
        use_rerank = rerank if rerank is not None else bool(getattr(slot.cfg, "rerank_default", False))
        return slot.retriever.search(
            query, kind=kind, limit=limit,
            focus_file=focus_file, focus_symbol=focus_symbol,
            rerank=use_rerank,
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
    def call_graph(name: str, depth: int = 2, min_confidence: float = 0.0,
                   root: RootSlug = DEFAULT) -> dict:
        """Forward + backward call graph for a function. depth in [1,5].
        min_confidence (0..1) drops CALLS edges resolved below that
        confidence (0.95 import map, 0.90 same module, 0.85 imported file,
        0.75 unique name, 0.55 import distance, 0.3-0.4 fuzzy)."""
        return _retriever(root).call_graph(name, depth=depth, min_confidence=min_confidence)

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
                min_confidence: float = 0.0, root: RootSlug = DEFAULT) -> dict:
        """Multi-hop graph walk from seed symbols. Returns the relevant
        subgraph in one call so the agent doesn't need to chain neighborhood
        lookups. seeds: list of Function/Class names. hops in [1,5].
        min_confidence filters CALLS / INHERITS hops by resolution confidence."""
        return _retriever(root).explore(seeds=seeds, hops=hops, limit=limit,
                                        min_confidence=min_confidence)

    @mcp.tool()
    def impact_of(target: str, depth: int = 3, limit: int = 50,
                  min_confidence: float = 0.0, root: RootSlug = DEFAULT) -> dict:
        """Blast radius of a file or symbol: transitive callers, importers,
        co-changed files, and tests. Use before refactoring or to scope a PR
        review. min_confidence (0..1) ignores low-confidence CALLS edges."""
        return _retriever(root).impact_of(target, depth=depth, limit=limit,
                                          min_confidence=min_confidence)

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
    def rules_for(file: str, root: RootSlug = DEFAULT) -> list[dict]:
        """Auto-attach rules for `file`: matches .cursor/rules/*.mdc by glob,
        plus AGENTS.md / CLAUDE.md as always-on. Compatible with the Cursor
        Rules ecosystem — drop in existing .mdc files and they work here."""
        return _retriever(root).rules_for(file)

    @mcp.tool()
    def cypher(query: str, limit: int = 100, root: RootSlug = DEFAULT) -> dict:
        """Run a READ-ONLY Cypher query against the graph. Rejects writes.
        Schema: nodes (File, Module, Class, Function, Variable, Community,
        Route, Tool); edges (CONTAINS, IMPORTS, CALLS {confidence, method},
        CALLS_CANDIDATE, INSTANTIATES, REFERENCES_, INHERITS, DECORATED_BY,
        SIMILAR_TO, CO_CHANGED_WITH, TESTS, MEMBER_OF, HANDLES).
        Use n.id / n.name / n.file. File node uses .path not .name.
        Use label(r) for the relationship type (type(r) does not exist)."""
        return _retriever(root).cypher(query, limit=limit)

    # --- analysis tools ---------------------------------------------------

    @mcp.tool()
    def context(symbol: str, file: str | None = None, tokens: int = 2000,
                min_confidence: float = 0.0, root: RootSlug = DEFAULT) -> dict:
        """One-call 360 view of a symbol trimmed to a token budget:
        definition, signature and doc, ranked callers and callees (PageRank
        x confidence), tests, flows it participates in (routes / MCP tools /
        entry points), its cluster, recent commits, matching rules and
        similar symbols. `text` is the ready-to-read Markdown; the lists are
        trimmed to what the text kept. Start here before editing a symbol."""
        return _retriever(root).context(symbol, file=file, tokens=tokens,
                                        min_confidence=min_confidence)

    @mcp.tool()
    def detect_changes(ref: str | None = None, diff: str | None = None, depth: int = 3,
                       min_confidence: float = 0.0, root: RootSlug = DEFAULT) -> dict:
        """Impact of a change set. Pass a unified `diff` text, or a git `ref`
        (None = working tree vs HEAD, 'HEAD' = last commit, 'main' = branch
        vs main, or a sha). Returns changed symbols, their transitive
        callers, affected flows (routes, tools, entry points), tests to run
        (with a pytest command) and an explainable 0-100 risk score."""
        return _retriever(root).detect_changes(ref=ref, diff=diff, depth=depth,
                                               min_confidence=min_confidence)

    @mcp.tool()
    def repo_map(focus: list[str] | None = None, tokens: int = 1024,
                 exclude_tests: bool = True, root: RootSlug = DEFAULT) -> dict:
        """Aider-style repository map: signatures only, ranked by PageRank
        personalized to `focus` (file paths and/or symbol names), shrunk by
        binary search to fit `tokens`. Use it to orient in an unfamiliar
        area without reading whole files."""
        return _retriever(root).repo_map(focus=focus, tokens=tokens, exclude_tests=exclude_tests)

    @mcp.tool()
    def list_clusters(limit: int = 100, root: RootSlug = DEFAULT) -> dict:
        """Communities detected at index time (Louvain over calls, types,
        containment and imports): id, auto-generated name, size, cohesion,
        top members and files, plus call counts between clusters."""
        return _retriever(root).list_clusters(limit=limit)

    @mcp.tool()
    def cluster(id: int | None = None, name: str | None = None, limit: int = 200,
                root: RootSlug = DEFAULT) -> dict:
        """One cluster by id (or name substring): members by PageRank, its
        API (members called from outside), the clusters it depends on and
        the clusters that use it."""
        return _retriever(root).cluster(id=id, name=name, limit=limit)

    @mcp.tool()
    def route_map(filter: str | None = None, limit: int = 500,
                  root: RootSlug = DEFAULT) -> dict:
        """HTTP routes (FastAPI, Flask, aiohttp, Starlette, Express) and MCP
        tools/resources/prompts found in the code, each with its handler
        function and how many functions it reaches. filter: substring of the
        route, handler or file."""
        return _retriever(root).route_map(filter=filter, limit=limit)

    @mcp.tool()
    def api_impact(route: str, depth: int = 4, min_confidence: float = 0.0,
                   root: RootSlug = DEFAULT) -> dict:
        """What one route or MCP tool touches: handler, reachable functions
        (by depth), files, tests covering them, and other routes sharing its
        dependencies. route: 'GET /api/x', a path, or a tool name."""
        return _retriever(root).api_impact(route, depth=depth, min_confidence=min_confidence)

    @mcp.tool()
    def trace(a: str, b: str, max_depth: int = 8, min_confidence: float = 0.0,
              root: RootSlug = DEFAULT) -> dict:
        """Shortest directed call path from symbol a to symbol b (CALLS plus
        class->method and function->instantiated-class member edges). If no
        forward path exists it tries b -> a and says so in `direction`."""
        return _retriever(root).trace(a, b, max_depth=max_depth, min_confidence=min_confidence)

    @mcp.tool()
    def health(limit: int = 15, min_confidence: float = 0.5,
               root: RootSlug = DEFAULT) -> dict:
        """Code health report: hubs (fan-in/out), bridges (approximate
        betweenness), dead code candidates, import cycles, large functions,
        and untested hotspots (high PageRank, no test reaches them)."""
        return _retriever(root).health(limit=limit, min_confidence=min_confidence)

    @mcp.tool()
    def symbol_history(name: str, file: str | None = None, limit: int = 10,
                       root: RootSlug = DEFAULT) -> dict:
        """When a symbol first appeared and last changed (from git blame at
        index time), its `git log -L` history, and matching symbols that were
        removed in earlier index runs."""
        return _retriever(root).symbol_history(name, file=file, limit=limit)

    @mcp.tool()
    def rename(symbol: str, new_name: str, file: str | None = None,
               include_text: bool = True, root: RootSlug = DEFAULT) -> dict:
        """PLAN a rename (never writes): definition and graph-confirmed
        reference lines (source 'graph') plus word-boundary text matches
        (source 'text', lower confidence), each with before/after. Applying
        is only possible through POST /api/rename with dry_run=false and
        apply=true."""
        return _retriever(root).rename_plan(symbol, new_name, file=file, include_text=include_text)

    # --- resources ----------------------------------------------------------
    # Default-root URIs plus root-scoped templates (docgraph://roots/<slug>/...).

    def _schema_text() -> str:
        return (
            "DocGraph schema (Kuzu)\n"
            "Nodes: File(path, language, lines, pagerank), Module(name), "
            "Class(name, qname, file, line_start, line_end, pagerank), "
            "Function(name, qname, file, line_start, line_end, signature, is_test, pagerank, "
            "first_seen_commit, last_changed_commit), Variable, Chunk, "
            "Community(name, size, cohesion), Route(name, method, path, framework, file, line), "
            "Tool(name, kind, framework, file, line)\n"
            "Edges: CONTAINS, IMPORTS, IMPORTS_SYMBOL, CALLS(line, confidence, method), "
            "CALLS_CANDIDATE(line, confidence, method), INSTANTIATES(confidence, method), "
            "INHERITS(confidence, method), OVERRIDES, DECORATED_BY, REFERENCES_, RETURNS, "
            "SIMILAR_TO(score), CO_CHANGED_WITH(count), TESTS, CONTAINS_CHUNK, LINKS_TO, "
            "MEMBER_OF(-> Community), HANDLES(Route/Tool -> Function)\n"
            "Confidence tiers: 1.0 scip, 0.95 import_map, 0.90 same_module, 0.85 import_suffix, "
            "0.75 unique_global, 0.55 import_distance, 0.40 common_name/fuzzy, 0.30 fuzzy_ranked.\n"
            "Cypher: label(r) not type(r); File uses .path; REFERENCES_ has a trailing underscore."
        )

    def _mermaid(root) -> str:
        data = _retriever(root).list_clusters(limit=40)
        lines = ["graph LR"]
        names = {}
        for c in data.get("clusters", []):
            nid = f"c{c['id']}"
            names[c["id"]] = nid
            label = str(c["name"]).replace('"', "'")
            lines.append(f'  {nid}["{label} ({c["size"]})"]')
        for l in data.get("links", [])[:120]:
            if l["src"] in names and l["dst"] in names:
                lines.append(f"  {names[l['src']]} -->|{l['count']}| {names[l['dst']]}")
        return "\n".join(lines)

    @mcp.resource("docgraph://schema", mime_type="text/plain")
    def res_schema() -> str:
        """Node and edge tables of the graph, with confidence tiers."""
        return _schema_text()

    @mcp.resource("docgraph://clusters", mime_type="application/json")
    def res_clusters() -> dict:
        """Clusters of the default root."""
        return _retriever(DEFAULT).list_clusters()

    @mcp.resource("docgraph://cluster/{cid}", mime_type="application/json")
    def res_cluster(cid: str) -> dict:
        """One cluster of the default root."""
        return _retriever(DEFAULT).cluster(id=int(cid))

    @mcp.resource("docgraph://flows", mime_type="application/json")
    def res_flows() -> list:
        """Execution flows (route / tool handlers and entry points)."""
        return _retriever(DEFAULT).processes(limit=40)

    @mcp.resource("docgraph://flow/{fid}", mime_type="application/json")
    def res_flow(fid: str) -> dict:
        """One flow by its entry function id."""
        return _retriever(DEFAULT).flow(int(fid)) or {"found": False}

    @mcp.resource("docgraph://routes", mime_type="application/json")
    def res_routes() -> dict:
        """HTTP routes and MCP tools with their handlers."""
        return _retriever(DEFAULT).route_map()

    def _root_of(slug: str):
        return RootSlug(slug)

    @mcp.resource("docgraph://roots/{slug}/clusters", mime_type="application/json")
    def res_root_clusters(slug: str) -> dict:
        """Clusters of a named root."""
        return _retriever(_root_of(slug)).list_clusters()

    @mcp.resource("docgraph://roots/{slug}/flows", mime_type="application/json")
    def res_root_flows(slug: str) -> list:
        """Flows of a named root."""
        return _retriever(_root_of(slug)).processes(limit=40)

    @mcp.resource("docgraph://roots/{slug}/routes", mime_type="application/json")
    def res_root_routes(slug: str) -> dict:
        """Routes of a named root."""
        return _retriever(_root_of(slug)).route_map()

    # --- prompts --------------------------------------------------------------

    @mcp.prompt()
    def detect_impact(ref: str = "", root: str = "") -> str:
        """Review the impact of the current change set before committing."""
        r = _root_of(root) if root else DEFAULT
        data = _retriever(r).detect_changes(ref=ref or None)
        risk = data.get("risk", {})
        syms = ", ".join(f"{s['name']} ({s['risk']['level']})" for s in data.get("changed_symbols", [])[:15])
        return (
            "You are reviewing a change set with DocGraph.\n"
            f"Overall risk: {risk.get('score')} ({risk.get('level')}): {risk.get('summary')}\n"
            f"Changed symbols: {syms or 'none indexed'}\n"
            f"Affected flows: {len(data.get('flows', []))}, callers: {len(data.get('callers', []))}\n"
            f"Tests to run: {data.get('test_command') or 'none found'}\n\n"
            "1. For each high or critical symbol, call `context` and check its callers.\n"
            "2. Run the tests above; if a changed symbol has no tests, say so.\n"
            "3. Summarise what could break, citing file:line."
        )

    @mcp.prompt()
    def architecture_map(root: str = "") -> str:
        """Explain the architecture from DocGraph clusters, with a Mermaid map."""
        r = _root_of(root) if root else DEFAULT
        return (
            "Explain this codebase's architecture using the cluster map below. For each "
            "cluster give its responsibility (use `cluster` for members) and the key "
            "dependencies between clusters. Keep the Mermaid diagram in your answer.\n\n"
            "```mermaid\n" + _mermaid(r) + "\n```"
        )

    return mcp
