# DocGraph — Working Notes for Claude

This file captures the things you can't infer from the code in 10 seconds. Read it before making non-trivial changes.

## What this project is

A local code knowledge graph backed by a single embedded Kuzu file. Indexes any repo with tree-sitter, embeds entities with fastembed, and exposes everything via 6 MCP tools and a single-page web UI.

It is the **v2 rewrite** of an older Neo4j + ChromaDB + Streamlit + Vite stack. The old code is preserved at tag `v1-legacy`. Do not resurrect any of those dependencies.

## Hard rules

- **One Python package, one process, one DB file.** No new top-level dirs, no separate frontend builds, no microservices.
- **Kuzu is the only data store.** Don't add SQLite, ChromaDB, Neo4j, Redis, etc.
- **No npm.** The web UI is one HTML file at `docgraph/ui/index.html`. No build step.
- **No torch dependency.** Embeddings go through `fastembed` (ONNX). Adding sentence-transformers/torch would 4× the install size.
- **Per-language processor classes are forbidden.** All language support comes from `parse.py::LANGUAGES` + `TAGS_QUERIES`. Adding a language = pip install `tree-sitter-<x>` and add two dict entries.

## File map

| File | Purpose |
|---|---|
| `docgraph/cli.py` | Typer entry. Add subcommands here. |
| `docgraph/config.py` | Auto-detects repo root (walks up for `.git`), respects `.gitignore` + `.docgraphignore`. |
| `docgraph/parse.py` | tree-sitter wrappers + tags queries per language. |
| `docgraph/index.py` | Parallel pipeline + per-file delta updates. **Most complex file.** |
| `docgraph/db.py` | Kuzu schema + bulk insert helpers. |
| `docgraph/embed.py` | fastembed wrapper (BGE-small, 384-dim). |
| `docgraph/rank.py` | NetworkX PageRank over CALLS+REFERENCES_+INHERITS+INSTANTIATES. |
| `docgraph/retrieve.py` | Hybrid retrieval. **All Cypher lives here or in `db.py`.** |
| `docgraph/mcp_tools.py` | 6 MCP tools. Keep this surface tight. |
| `docgraph/server.py` | FastAPI: web UI + JSON API. |
| `docgraph/ui/index.html` | Single-page graph viewer. Vanilla JS, canvas, no deps. |

Runtime data: `<repo>/.docgraph/graph.kuzu/` (DB) and `<repo>/.docgraph/cache.json` (per-file `{hash, entities, edges}` for delta updates).

## Kuzu Cypher gotchas (learned the hard way)

- Use `label(r)` for relationship type. **`type(r)` does not exist** in Kuzu.
- `startNode(r)` and `endNode(r)` **do not exist**. Use `(a)-[r]->(b)` and reference `a` / `b` directly.
- `nodes(path)` works for variable-length paths; `relationships(path)` does not on all versions — avoid it.
- Property name on `File` nodes is `path`, not `name`. On all other entity nodes it's `name`. Easy to confuse in `MATCH` clauses.
- `REFERENCES_` (with trailing underscore) — `REFERENCES` is a reserved word in Kuzu.
- Edge tables can declare multiple FROM/TO pairs in one statement (see `db.py::EDGE_DDL`); use that instead of separate edge tables per type-combination.
- Bulk insert: `UNWIND $rows AS row CREATE (n:Label {col: row.col, ...})` is much faster than per-row CREATE. The helpers in `db.py::insert_nodes` / `insert_edges` already do this.
- A reader connection (`read_only=True`) holds a lock that blocks writers. The MCP server and the web server both open read-only — kill them before running `docgraph index` or you'll get `"Could not set lock on file"`.

## Tree-sitter API gotchas

- We're on `tree-sitter >= 0.25`. The query API changed: use `ts.Query(lang, src)` and `ts.QueryCursor(q).captures(node)` — **not** `lang.query(...).captures(...)` from older docs.
- `captures()` returns `dict[str, list[Node]]`, not the old `list[tuple[Node, str]]`.
- Each language is its own pip package (`tree-sitter-python`, `tree-sitter-typescript`, etc.). The TS package exposes `language_typescript()` and `language_tsx()` (two grammars in one package).
- The `tree-sitter-language-pack` package looks tempting but it's a Rust-native rewrite (`_native` module) that needs runtime grammar downloads from the network. Don't switch to it.

## ID allocation across runs

The indexer uses an integer `id` primary key per entity. On full reindex it starts at 1. On incremental it must continue from `max(id) + 1` in the DB — `_seed_ids_from_db()` handles this. Don't break this invariant; Kuzu rejects duplicate primary keys with a confusing error.

## Per-file delta correctness

The trickiest part of the codebase. Read `index.py::index_all` carefully before changing it. The contract:

1. Hash-compare to bucket files into changed / added / deleted / unchanged.
2. `DETACH DELETE` changed + deleted file nodes — Kuzu removes incident edges in the same step.
3. Re-parse only changed + added files. Update cache.
4. Build the symbol table from current DB state (not from the parse output).
5. For each cached `RawEdge` across all files: insert only if `needs_insert(src_file, target_file)` — i.e., at least one endpoint was just (re)created. Edges fully inside the unchanged set are still in the DB and skipping them prevents duplicates.
6. Tier 4 edges (`SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`) and PageRank are wiped and recomputed every time. They're global and cheap.

If you change the parse output shape, update the cache writer **and** the cache reader in lockstep.

## Testing the indexer after changes

There's no formal test suite yet. Smoke test by:

```bash
# 1. Clean baseline
rm -rf .docgraph
.venv/Scripts/docgraph index --full
.venv/Scripts/docgraph stats > /tmp/full_stats.txt

# 2. Touch + edit + delete cycle
echo "" >> docgraph/embed.py            # content edit
.venv/Scripts/docgraph index            # should be ~1.3s
git checkout docgraph/embed.py          # revert
.venv/Scripts/docgraph index            # ~1.3s, stats back to baseline
.venv/Scripts/docgraph stats > /tmp/incremental_stats.txt

# 3. Diff — should be identical
diff /tmp/full_stats.txt /tmp/incremental_stats.txt
```

If the diff is non-empty, your change broke incremental correctness.

## MCP tool surface — keep it tight

6 tools, no more without a strong reason: `search`, `definition`, `references`, `call_graph`, `file_map`, `neighborhood`. The whole point is "fewer, sharper tools than the competition."

`neighborhood` is the differentiator — it ranks related code via `CALLS + REFERENCES_ + SIMILAR_TO + INHERITS + TESTS` together, ordered by PageRank. It answers "what else should I read to understand this?"

## Coding conventions

- Type-hint everything. Use `from __future__ import annotations` at the top of files that use `|` union syntax for older Python compat.
- Cypher strings: keep them in `db.py` or `retrieve.py`. Don't scatter them across `mcp_tools.py` / `server.py`.
- Python 3.10 is the floor. Don't use 3.12+ syntax.
- No emojis in code or commit messages unless the user explicitly asks. The user runs Windows + cp1252 console; Unicode in tool descriptions has crashed `await mcp.call_tool` in the past (see commit history for the `∈` incident).
- One short comment max above non-obvious blocks. Don't narrate.

## Performance targets

These should not regress:

| Scenario | Target |
|---|---|
| No-op incremental | < 0.05s |
| Touch (same hash) | < 0.05s |
| 1-file edit | < 2s |
| Full index, 100k LOC | < 10s on a modern laptop |
| MCP tool call | < 200ms typical |

Anything that pushes past these needs a comment explaining why.

## Common dev commands

```bash
.venv/Scripts/pip install -e .                 # editable install
.venv/Scripts/docgraph index --full            # rebuild from scratch
.venv/Scripts/docgraph serve                   # http://127.0.0.1:5500
.venv/Scripts/docgraph stats                   # quick health check

# Kill any running servers/MCPs holding the DB lock:
taskkill //F //IM python.exe                   # Windows
pkill -f docgraph                              # *nix
```

## Things that have broken before — don't repeat

- Putting `∈` or other non-cp1252 chars in MCP tool docstrings → crashes the call on Windows.
- Calling `type(r)` in Cypher → `function TYPE does not exist`.
- Forgetting that `File.path` is the name property (not `File.name`).
- Re-running the indexer while `docgraph serve` is running → DB lock error.
- Using `tree-sitter-language-pack 1.6+` thinking it's the old Goldziher API — it isn't, it's a Rust rewrite that needs network downloads.
- Inserting nodes with IDs starting at 1 on an incremental run → duplicate-PK error from Kuzu. Always `_seed_ids_from_db()` first.

## Known limitations / next-up

- Embedding model loads fresh per process (~1s cold). Pre-warming or caching across CLI invocations would help.
- No `docgraph watch` mode yet.
- Force-directed canvas renderer is O(N²) per frame; fine to ~2k nodes. For larger graphs, drop in Cosmograph (WebGL) at `docgraph/ui/index.html`.
- `IMPORTS_SYMBOL` and `OVERRIDES` edges declared in the schema but not extracted by any tags query yet.
- Cross-repo references (monorepo) are not modeled — every `.docgraph/` is per-repo.
