# DocGraph

Local code knowledge graph for LLMs. Indexes any repo, exposes it via MCP, and ships an interactive graph UI.

- **One file embedded DB** (Kuzu) — no Neo4j, no ChromaDB.
- **All 165+ languages on day one** via tree-sitter.
- **Parallel indexer** — ~5 sec for 100k LOC, sub-second incremental.
- **Live graph UI** — force-directed, semantic clustering, click-to-inspect.
- **MCP server** — 6 tools, stdio for Cursor/Claude or HTTP for web.
- **Tier 4 differentiator edges**: SIMILAR_TO (vector), CO_CHANGED_WITH (git history), TESTS (heuristic).

## Install

```bash
pipx install docgraph        # or: pip install docgraph
```

## Usage

```bash
cd /your/repo
docgraph index               # parallel index, incremental on subsequent runs
docgraph serve               # http://127.0.0.1:5500 — graph UI
docgraph mcp                 # stdio MCP for Cursor/Claude
docgraph stats               # entity + edge counts
```

## MCP install (Cursor / Claude Desktop)

```bash
docgraph install-mcp
```

Copy the printed JSON into your client's MCP config.

## Relationships extracted

| Tier | Edges |
|---|---|
| Structural | CONTAINS, IMPORTS, IMPORTS_SYMBOL |
| Behavioral | CALLS, INSTANTIATES, REFERENCES_, RETURNS |
| Type system | INHERITS, IMPLEMENTS, OVERRIDES, DECORATED_BY |
| Differentiators | SIMILAR_TO, CO_CHANGED_WITH, TESTS |

## MCP tools

- `search(query, kind?, limit?)` — hybrid vector + name + PageRank search
- `definition(name, file?)` — full body of a symbol
- `references(name)` — all callers/users
- `call_graph(name, depth=2)` — forward + backward
- `file_map(file)` — entities + imports in a file
- `neighborhood(name, limit=10)` — PageRank-ranked related code (calls + similarity + tests + inheritance)

## Architecture

```
docgraph/
  cli.py          # typer entry
  config.py       # auto-detect repo root + ignores
  parse.py        # tree-sitter universal parser
  index.py        # parallel pipeline (walk → parse pool → embed → bulk write)
  db.py           # Kuzu schema + bulk insert helpers
  embed.py        # fastembed wrapper (BGE-small, ONNX)
  rank.py         # PageRank over call graph
  retrieve.py     # hybrid retrieval
  mcp_tools.py    # 6 MCP tools
  server.py       # FastAPI: web UI + JSON API
  ui/index.html   # single-page Cosmograph-style viewer
```

Data lives at `<repo>/.docgraph/graph.kuzu`.

## License

MIT
