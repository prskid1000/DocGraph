# DocGraph

A local code knowledge graph for LLMs. Indexes any repo with tree-sitter, stores entities and relationships in a single embedded Kuzu file, and exposes everything via MCP and a live web graph UI.

```bash
pipx install docgraph
cd /any/repo
docgraph index           # ~5s for 100k LOC; sub-second incremental
docgraph serve           # http://127.0.0.1:5500
docgraph mcp             # stdio MCP for Cursor / Claude Desktop
```

## Why

Most code-intelligence tools either ship a heavy multi-service stack (Neo4j + a vector DB + a separate UI app) or a thin keyword search. DocGraph keeps everything in one Python package backed by one file:

- **One file embedded DB** (Kuzu) — graph + vectors, no servers.
- **165+ languages** out of the box via tree-sitter (just install more `tree-sitter-*` packages).
- **Parallel indexer** — process pool, batched embeddings, bulk Cypher writes.
- **Per-file delta updates** — sub-second on edits, 0ms on no-op runs.
- **Live graph UI** — single HTML file, force-directed canvas, no npm build.
- **MCP server** — 6 tight tools, stdio for editors or HTTP for web clients.
- **Differentiator edges** — `SIMILAR_TO` (vector top-K), `CO_CHANGED_WITH` (git history), `TESTS` (heuristic). The "what else will my change break?" answer.

## Performance

| Scenario | Time |
|---|---|
| Full index, ~13 files | 6.2s (cold model load) |
| No-op incremental | 0.01s |
| Touch only (same hash) | 0.00s |
| 1-file content edit | 1.3s |
| New file added | 1.3s |
| File deleted | 0.23s (no parse needed) |

Incremental and full produce identical stats — verified by add/edit/delete cycles.

## Install

```bash
pipx install docgraph    # recommended; isolated install
# or
pip install docgraph
```

Requires Python 3.10+. The first run downloads the embedding model (~30 MB BGE-small-en, ONNX).

## CLI

| Command | What it does |
|---|---|
| `docgraph index [path]` | Parallel index. Incremental by default; `--full` to wipe and rebuild |
| `docgraph serve [path]` | Start the web UI + JSON API on port 5500 |
| `docgraph mcp [path]` | Run MCP server (stdio default; `--transport http` for HTTP) |
| `docgraph stats [path]` | Print entity + edge counts |
| `docgraph clear [path]` | Delete `.docgraph/` for the repo |
| `docgraph install-mcp [path]` | Print the JSON snippet for Cursor / Claude Desktop |
| `docgraph version` | Print version |

`path` defaults to the current directory; the repo root is auto-detected by walking up to find `.git`.

## MCP install (Cursor / Claude Desktop)

```bash
docgraph install-mcp
```

Copy the printed JSON into your client's MCP config. Example for Claude Desktop:

```json
{
  "mcpServers": {
    "docgraph-myrepo": {
      "command": "docgraph",
      "args": ["mcp", "/absolute/path/to/repo"]
    }
  }
}
```

## MCP tools

| Tool | What it returns |
|---|---|
| `search(query, kind?, limit=10)` | Hybrid vector + name + PageRank ranked results |
| `definition(name, file?)` | Full body + metadata of a symbol |
| `references(name)` | All callers / usages |
| `call_graph(name, depth=2)` | Forward + backward call graph (depth 1–5) |
| `file_map(file)` | Entities + outgoing imports for a file |
| `neighborhood(name, limit=10)` | PageRank-ranked related code via calls + similarity + tests + inheritance — the "what else should I read?" tool |

## Relationships extracted

| Tier | Edges |
|---|---|
| **Structural** | `CONTAINS`, `IMPORTS`, `IMPORTS_SYMBOL` |
| **Behavioral** | `CALLS`, `INSTANTIATES`, `REFERENCES_`, `RETURNS` |
| **Type system** | `INHERITS`, `IMPLEMENTS`, `OVERRIDES`, `DECORATED_BY` |
| **Differentiators** | `SIMILAR_TO` (vector top-K), `CO_CHANGED_WITH` (git history), `TESTS` (heuristic name match) |

Nodes: `File`, `Module`, `Class`, `Function`, `Variable`. Each `Function` and `Class` carries an embedding and a PageRank score.

## Languages bundled

Out of the box: **python, javascript, typescript, tsx, java, go, rust, c, cpp, c_sharp, ruby, php, bash, html, css, json, yaml**.

Adding more is two steps:

```bash
pip install tree-sitter-<lang>
```

then add an entry to `LANGUAGES` and a query to `TAGS_QUERIES` in `docgraph/parse.py`. Most grammars work with the standard `(function_definition name: (_) @name) @definition.function` shape.

## Architecture

```
docgraph/
  cli.py          # typer entry: index / serve / mcp / stats / clear / install-mcp
  config.py       # auto-detect repo root, .gitignore, .docgraphignore
  parse.py        # tree-sitter universal parser (per-language tags queries)
  index.py        # parallel pipeline + per-file delta updates
  db.py           # Kuzu schema (5 node tables, 13 edge tables) + bulk insert
  embed.py        # fastembed wrapper (BGE-small ONNX, 384-dim)
  rank.py         # PageRank over call + reference + inheritance graph
  retrieve.py     # hybrid retrieval (vector cosine + name boost + PageRank)
  mcp_tools.py    # 6 MCP tools wrapping the retriever
  server.py       # FastAPI: web UI + JSON API
  ui/index.html   # single-page force-directed canvas viewer (zero deps)
```

Data lives at `<repo>/.docgraph/`:
- `graph.kuzu/` — the embedded DB
- `cache.json` — per-file `{hash, entities, edges}` for delta updates

## How incremental works

1. Walk repo, SHA1-hash each file, compare to cache.
2. Bucket files into **changed / added / deleted / unchanged**.
3. `DETACH DELETE` only changed/deleted files' nodes — Kuzu drops incident edges in the same step.
4. Re-parse only changed/added files in a process pool.
5. Continue ID allocation from `max(id) + 1` in the DB.
6. Re-resolve only edges that touch a changed file; edges fully inside the unchanged set stay untouched.
7. Recompute `SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`, and PageRank — they're global and cheap.

## JSON API (when running `docgraph serve`)

| Endpoint | Notes |
|---|---|
| `GET /` | The web UI |
| `GET /api/search?q=...&kind=...&limit=10` | Same as the MCP tool |
| `GET /api/definition?name=...&file=...` | |
| `GET /api/references?name=...` | |
| `GET /api/call_graph?name=...&depth=2` | |
| `GET /api/file_map?file=...` | |
| `GET /api/neighborhood?name=...&limit=10` | |
| `GET /api/graph?limit_nodes=2000` | All nodes + edges for the viewer |
| `GET /api/stats` | Entity counts + table list |
| `GET /api/file_content?file=...` | Source text for inspection (sandboxed to repo root) |

## Comparison

| | DocGraph | GitNexus | Codebase-Memory | Cursor |
|---|---|---|---|---|
| Embedded DB | Kuzu (graph + vectors) | KuzuDB / LadybugDB | SQLite | proprietary |
| Languages day 1 | 17, easy to add | many | 66 | many |
| Graph UI | live force-directed | Mermaid (static) | none | none |
| Per-file delta | ✅ | ✅ | ✅ | ✅ |
| `SIMILAR_TO` edge | ✅ | (implicit) | ❌ | implicit |
| `CO_CHANGED_WITH` edge | ✅ | ❌ | ❌ | ❌ |
| `TESTS` edge | ✅ | ❌ | ❌ | ❌ |
| MCP tools | 6 | 7 | 14 | n/a |
| Install | `pipx install` | manual | manual | proprietary IDE |

## Roadmap

- Pre-download / cache embedding model so cold start is sub-second.
- Optional `query_graph` MCP tool for raw Cypher (power users).
- Watcher mode (`docgraph watch`) auto-reindexes on file changes.
- Cross-repo links for monorepo / multi-repo setups.
- Sigma.js / Cosmograph backend for >5k node graphs.

## License

MIT
