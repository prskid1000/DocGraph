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

Most code-intelligence tools either ship a heavy multi-service stack (Neo4j + a vector DB + a separate UI app) or a thin keyword search. DocGraph keeps everything in one Python package backed by one file.

### Architecture

- **One file embedded DB** (Kuzu) — graph + vectors, no servers.
- **165+ languages** out of the box via tree-sitter (just install more `tree-sitter-*` packages).
- **Parallel indexer** — process pool, batched embeddings, bulk Cypher writes.
- **Per-file delta updates** — sub-second on edits, 0 ms on no-op runs.
- **Optional GPU acceleration** — `docgraph index --gpu` runs embeddings via ONNX Runtime on CUDA / DirectML / CoreML for a multi-x speedup on large repos. Still no torch dep — just `pip install onnxruntime-gpu` (NVIDIA) or `onnxruntime-directml` (Windows). Falls back to CPU silently if no GPU runtime is installed.
- **Local-only by default** — no telemetry, no cloud round-trips. The only outbound network calls are opt-in: `docgraph docs add <url>` (you supply the URL) and `--llm-docstrings` (you supply the local server).

### Retrieval

- **MCP server** — 15 tools (6 base + 9 differentiators). Two transports: `stdio` for editors (Cursor / Claude Desktop) and `http` for web clients (`docgraph mcp --transport http`).
- **Differentiator edges** — `SIMILAR_TO` (vector top-K), `CO_CHANGED_WITH` (git history), `TESTS` (heuristic). The "what else will my change break?" answer.
- **Differentiator MCP tools** — `explore` (multi-hop BFS), `impact_of` (blast radius), `test_impact` (which tests cover this?), and `cypher` (raw read-only graph query — rejects `CREATE` / `MERGE` / `SET` / `DELETE` / `DROP` server-side, so agents can't accidentally mutate the graph).
- **Personalized PageRank** — `search` accepts `focus_file` / `focus_symbol` and ranks results by proximity to the file or symbol the agent is currently editing.
- **Cross-encoder reranker** — opt-in `search(rerank=True)` lifts top-K precision via a 33 MB Jina cross-encoder (still local, still ONNX).
- **Scope-aware resolution** — `CALLS` / `INSTANTIATES` / `INHERITS` prefer same-file then imported-file targets, killing most overload hallucinations without an LSP daemon.
- **Sub-function chunking** — long bodies split + embedded per chunk; search max-pools across chunks so a 1000-line class still has fine recall.
- **Diff- and history-aware tools** — `git_changes` (changed entities + 1-hop callers — Cursor `@Commit` joined to the graph), `git_blame` (line-range blame — Cursor `@Blame` parity), `git_recent` (last N commits scoped to a file or repo).

### Watcher + UI

- **Live graph UI** — single HTML file, force-directed canvas, no npm build. Sigma.js WebGL engine auto-engages above 2 k nodes.
- **Watcher** — `docgraph watch` auto-reindexes on file changes (Rust `notify` under the hood, debounced).
- **Live UI auto-redraw** — `docgraph watch --serve` runs the watcher and the web server in **one process** so they share the Kuzu file lock. After every reindex the browser refreshes itself via Server-Sent Events at `/api/events` — no F5, no polling.
- **ML-training-style progress bars** — every index phase (parse, embed entities, embed chunks, write nodes, build symbol table, resolve edges, `SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`, PageRank, persist) reports `% | M/N | elapsed | ETA`. Same bars in `docgraph docs add`.

### Multi-repo + ignores

- **Multi-repo** — `--repo` (repeatable) merges several repos into one graph; cross-repo `IMPORTS` resolve naturally. List persisted in `.docgraph/repos.json`.
- **Smart default ignores** — universal baseline (Cursor-parity: `node_modules/`, `__pycache__/`, `.venv/`, `.next/`, `.gradle/`, lockfiles, binaries, plus Jupyter / MLflow / wandb / DVC / R / Haskell / Zig caches) layered with per-ecosystem autodetect (Node / Python / Maven / Gradle / Rust / .NET / Angular / Android / Swift / Ruby / Dart / Elixir / Scala / PHP / Go / Terraform / Unity) — ambiguous build dirs (`target/`, `build/`, `bin/`, `obj/`) only ignored when their marker file is detected.
- **Two-tier ignore** — `.cursorindexingignore` skips files entirely; `.cursorignore` indexes them but redacts bodies/snippets returned to the AI. The HTTP API also sandboxes `/api/file_content` to the repo root (403 on traversal) and redacts `.cursorignore`'d files.
- **Cursor-rules compatible** — drops in existing `.cursor/rules/*.mdc` and `AGENTS.md`; exposes them via `rules_for(file)` so any MCP client gets glob-matched auto-attach.

### Optional augmentation

- **`@Docs` ingestion** — `docgraph docs add <url>` fetches and embeds external API docs; `search_docs(query)` MCP tool surfaces them. Idempotent (re-ingesting a URL replaces prior chunks).
- **Optional LLM-augmented docstrings** — opt-in via `--llm-docstrings`; talks to any OpenAI- or Anthropic-compatible local server (LM Studio, llama.cpp, vLLM, Ollama). One-sentence summaries for entities lacking native docs lift retrieval recall on under-documented codebases. Cached by body hash so incrementals stay fast.

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

**Optional GPU acceleration** — install one of these alongside `docgraph` to enable `--gpu` (no torch dep, all ONNX):

```bash
pip install onnxruntime-gpu          # NVIDIA / CUDA (Linux, Windows)
pip install onnxruntime-directml     # Windows / any GPU (DirectX 12)
pip install onnxruntime-silicon      # Apple Silicon (CoreML)
```

DocGraph picks whichever provider is installed automatically; without one it stays on CPU.

## CLI reference

`path` argument defaults to the current directory; the repo root is auto-detected by walking up to find `.git`.

### `docgraph index [path]`

Parallel index. Incremental by default; pass `--full` to wipe and rebuild.

| Flag | Default | Description |
|---|---|---|
| `--full`, `-f` | `false` | Wipe the DB and rebuild from scratch instead of delta-update |
| `--repo PATH`, `-r PATH` | — | Additional repo root to include (repeatable). Persisted in `.docgraph/repos.json`; subsequent `watch` / `serve` / `mcp` pick it up automatically |
| `--llm-docstrings` | `false` | Generate one-sentence docstrings for entities lacking native docs via a local OpenAI/Anthropic-compatible LLM. Cached by body hash in `.docgraph/llm_docstrings.json` so incrementals don't re-call. |
| `--llm-port INT` | `1235` | Local LLM server port (host is always `localhost`) |
| `--llm-model STR` | `local-model` | Model name sent to the server (most local servers ignore this) |
| `--llm-format STR` | `openai` | API format: `openai` (Chat Completions @ `/v1/chat/completions`) or `anthropic` (Messages @ `/v1/messages`) |
| `--gpu` | `false` | Use GPU for embeddings via ONNX Runtime (CUDA / DirectML / CoreML / ROCm). Requires `onnxruntime-gpu`, `onnxruntime-directml`, or `onnxruntime-silicon` to be installed. Falls back to CPU silently if no GPU runtime is found. |
| `--verbose`, `-v` | `false` | Verbose logs |

### `docgraph watch [path]`

Auto-reindex on file changes (Rust `notify` under the hood, debounced). Plain `watch` holds a writer lock — kill `serve` / `mcp` against the same DB first. With `--serve`, the watcher and the web UI run in **one process**, sharing the DB lock; the browser stays in sync via Server-Sent Events.

| Flag | Default | Description |
|---|---|---|
| `--debounce INT` | `500` | Debounce window in ms before reindex fires |
| `--serve` | `false` | Also run the web UI + JSON API in the same process. UI auto-redraws after each reindex via `/api/events` SSE. |
| `--host STR` | `127.0.0.1` | Bind address (only with `--serve`) |
| `--port INT` | `5500` | Bind port (only with `--serve`) |
| `--verbose`, `-v` | `false` | Verbose logs |

### `docgraph serve [path]`

Start the web UI + JSON API.

| Flag | Default | Description |
|---|---|---|
| `--host STR` | `127.0.0.1` (or `$DOCGRAPH_HOST`) | Bind address |
| `--port INT` | `5500` (or `$DOCGRAPH_PORT`) | Bind port |
| `--verbose`, `-v` | `false` | Verbose access logs |

### `docgraph mcp [path]`

Run the Model Context Protocol server.

| Flag | Default | Description |
|---|---|---|
| `--transport STR` | `stdio` | `stdio` (for Cursor / Claude Desktop) or `http` (for web clients) |
| `--verbose`, `-v` | `false` | Verbose logs |

### `docgraph stats [path]`

Print entity + edge counts. No flags.

### `docgraph clear [path]`

Delete `.docgraph/` for the repo (DB + cache + repos list).

| Flag | Default | Description |
|---|---|---|
| `--yes`, `-y` | `false` | Skip the confirmation prompt |

### `docgraph install-mcp [path]`

Print a JSON snippet ready to paste into Cursor / Claude Desktop's MCP config. No flags.

### `docgraph docs add <url>`

Fetch a URL, chunk + embed it, store as `Doc` nodes for `search_docs`. Idempotent: re-adding the same URL deletes prior chunks first.

| Flag | Default | Description |
|---|---|---|
| `--path PATH` | cwd | Repo whose `.docgraph/` to write the doc nodes into |

### `docgraph docs list`

Show ingested doc URLs and their chunk counts.

| Flag | Default | Description |
|---|---|---|
| `--path PATH` | cwd | Repo to query |

### `docgraph docs remove <url>`

Delete all chunks for a previously-ingested doc URL.

| Flag | Default | Description |
|---|---|---|
| `--path PATH` | cwd | Repo to operate on |

### `docgraph version`

Print version. No flags.

### Environment variables

| Var | Used by | Default |
|---|---|---|
| `DOCGRAPH_HOST` | `serve`, `mcp` (http) | `127.0.0.1` |
| `DOCGRAPH_PORT` | `serve`, `mcp` (http) | `5500` |
| `DOCGRAPH_EMBED_MODEL` | `index` | `BAAI/bge-small-en-v1.5` |
| `DOCGRAPH_GPU` | `index`, `serve`, `mcp`, `watch`, `docs add` | unset (off). Set to `1`/`true` to use GPU for embeddings via ONNX Runtime. |
| `DOCGRAPH_LLM_DOCSTRINGS` | `index` | unset (off). Set to `1`/`true` to enable. |
| `DOCGRAPH_LLM_HOST` | `index` | `localhost` |
| `DOCGRAPH_LLM_PORT` | `index` | `1235` |
| `DOCGRAPH_LLM_MODEL` | `index` | `local-model` |
| `DOCGRAPH_LLM_FORMAT` | `index` | `openai` |
| `DOCGRAPH_LLM_API_KEY` | `index` | unset. If set, sent as `Authorization: Bearer …` (OpenAI) or `x-api-key: …` (Anthropic). |
| `DOCGRAPH_LLM_MAX_TOKENS` | `index` | `150` |
| `DOCGRAPH_LLM_TIMEOUT` | `index` | `30` (seconds) |

CLI flags override env vars. Env vars set defaults that survive across invocations.

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
| `search(query, kind?, limit=10, focus_file?, focus_symbol?)` | Hybrid vector + name + PageRank. Pass a focus → personalized PageRank biases ranking toward what the agent is reading |
| `definition(name, file?)` | Full body + metadata of a symbol |
| `references(name)` | All callers / usages |
| `call_graph(name, depth=2)` | Forward + backward call graph (depth 1–5) |
| `file_map(file)` | Entities + outgoing imports for a file |
| `neighborhood(name, limit=10)` | PageRank-ranked related code via calls + similarity + tests + inheritance — the "what else should I read?" tool |
| `explore(seeds, hops=3, limit=25)` | Multi-hop BFS subgraph from one or more seed names. Replaces chained `neighborhood` calls |
| `impact_of(target, depth=3)` | Blast radius: transitive callers, importers, co-changed files, and tests for a file or symbol |
| `test_impact(target)` | Tests that exercise `target` (file or symbol) via TESTS + reverse `CALLS*` |
| `cypher(query, limit=100)` | Read-only Cypher escape hatch for power agents. Rejects writes, caps rows |
| `git_changes(ref?)` | Diff-aware retrieval. `ref`: None (working tree), `HEAD`, `main` (branch vs main), `<sha>`. Returns files + entities + 1-hop callers. Mirrors Cursor `@Commit` / `@PR` / `@Recent Changes` |
| `git_blame(file, line_start, line_end?)` | `git blame` per line. Mirrors Cursor Blame |
| `git_recent(file?, limit=20)` | Recent commits, optionally scoped to a file |
| `rules_for(file)` | Auto-attach rules for a file: matches `.cursor/rules/*.mdc` by glob, plus `AGENTS.md` / `CLAUDE.md` always-on. Drop in existing Cursor `.mdc` rules and they work here |
| `search_docs(query, limit=10)` | Semantic search across ingested external docs (`docgraph docs add <url>`). Cursor `@Docs` parity |
| `search(..., rerank=True)` | Cross-encoder rerank (Jina tiny, ~33 MB) over the top candidates for token-level precision. Opt-in; first call downloads the model |

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
  mcp_tools.py    # 15 MCP tools wrapping the retriever
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
| `GET /api/file_content?file=...` | Source text for inspection (sandboxed to repo root; redacts `.cursorignore`'d files) |
| `GET /api/git_changes?ref=...` | Diff-aware retrieval |
| `GET /api/git_blame?file=...&line_start=&line_end=` | `git blame` |
| `GET /api/git_recent?file=...&limit=` | Recent commits |
| `GET /api/rules_for?file=...` | Auto-attach rules matching the file |
| `GET /api/search_docs?q=...&limit=` | Semantic search over ingested external docs |
| `GET /api/explore?seeds=a,b&hops=&limit=` | Multi-hop subgraph from seeds |
| `GET /api/impact_of?target=...&depth=&limit=` | Blast radius |
| `GET /api/test_impact?target=...&limit=` | Tests covering target |
| `POST /api/cypher` (`{query, limit}`) | Read-only Cypher |
| `GET /api/events` | Server-Sent Events stream. Emits `reindex_done` after every reindex when running under `docgraph watch --serve`; the bundled UI uses it to auto-refresh. Sends keepalive comments every 15 s. |

## Comparison

| | DocGraph | GitNexus | Codebase-Memory | Cursor | Greptile | Sourcegraph (Cody) | Continue.dev |
|---|---|---|---|---|---|---|---|
| License | MIT | open | open | proprietary | proprietary SaaS | Apache 2 (self-host) | Apache 2 |
| Runs fully local | ✅ | ✅ | ✅ | partial (cloud-augmented) | ❌ (cloud only) | ✅ (self-hosted) | ✅ |
| Embedded store | Kuzu (graph + vectors) | KuzuDB / LadybugDB | SQLite | proprietary | cloud | Postgres + cloud index | LanceDB / SQLite |
| Languages day 1 | 17 (tree-sitter) | many | 66 | many | many | many (SCIP) | many (tree-sitter) |
| Live graph UI | force-directed + Sigma WebGL | Mermaid (static) | ❌ | ❌ | ❌ | partial (call-graph view) | ❌ |
| Per-file incremental | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Personalized PageRank | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cross-encoder rerank | ✅ (opt-in) | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `SIMILAR_TO` edge | ✅ | implicit | ❌ | implicit | implicit | ❌ | implicit |
| `CO_CHANGED_WITH` edge | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `TESTS` edge | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Diff-aware retrieval | ✅ (`git_changes`) | ❌ | ❌ | partial (`@Commit`) | ❌ | ❌ | ❌ |
| Cursor-rules ingest | ✅ (`.mdc` + `AGENTS.md`) | ❌ | ❌ | native | ❌ | ❌ | ❌ |
| Optional LLM docs | ✅ (local OpenAI/Anthropic-compat) | ❌ | ❌ | ❌ | ✅ (cloud) | ❌ | partial |
| Live UI auto-redraw | ✅ (SSE on reindex) | ❌ | ❌ | n/a (IDE) | n/a | ❌ | n/a |
| Read-only Cypher escape hatch | ✅ | ❌ | ❌ | ❌ | ❌ | partial (GraphQL) | ❌ |
| MCP tools | 15 | 7 | 14 | n/a (IDE) | yes | via plugins | via plugins |
| Install | `pipx install` | manual | manual | proprietary IDE | hosted SaaS | self-host stack | binary |

> Aider, Bloop, OpenGrok, and CodeQL are adjacent but solve different problems (in-terminal pair-programming, semantic code search without graph, security analysis) and are omitted to keep the table focused on local code-graph tools.

## Multi-repo

```bash
docgraph index --repo /path/to/repo-b --repo /path/to/repo-c
docgraph watch       # picks up all repos automatically (persisted in .docgraph/repos.json)
docgraph serve
```

In multi-repo mode, file paths are prefixed with each repo's basename (`repo-b/src/foo.py`) so they stay unique. Cross-repo `IMPORTS` resolve naturally through the existing fuzzy import matcher.

## Tests

```bash
pip install pytest
pytest                   # ~26s (one-time embedder load + 162 tests)
```

Covers indexer correctness, per-file delta updates, all retrieval methods, every MCP tool (registered + invoked end-to-end), every HTTP API route (incl. `.cursorignore` redaction + cypher write-blocker), multi-repo walking, watch filter logic, and the embedding-text builder.

## Roadmap

Recently shipped (kept here for context):

- ✅ **Optional LLM-generated docstrings** — `--llm-docstrings`, talks to any local OpenAI/Anthropic-compatible server (LM Studio, llama.cpp, vLLM, Ollama). Cached by body hash so incrementals don't re-call.
- ✅ **Live UI auto-redraw on reindex** — `docgraph watch --serve` runs the watcher and the web UI in one process; the browser refreshes via Server-Sent Events at `/api/events` after each reindex.
- ✅ **Per-ecosystem default ignores** — autodetect Node/Python/Maven/Gradle/Rust/.NET/etc. and apply curated ignore patterns; ML/data-science cache dirs (`.ipynb_checkpoints/`, `mlruns/`, `wandb/`, `.dvc/cache/`) baked in universally.
- ✅ **ML-training-style progress bars** — every phase (parse, embed, write, similarity, PageRank, persist) reports % + M/N + elapsed + ETA.

In flight / queued:

- Precise (compiler-grade) symbol resolution via SCIP / LSP daemons (scope-aware-via-imports already covers the common case; LSP eliminates overload mis-resolution).
- Pre-download / cache embedding model so cold start is sub-second instead of ~1 s.
- Symbol-level imports (`IMPORTS_SYMBOL`) and method `OVERRIDES` edges — schema reserves them, parser hasn't extracted them yet.
- Cross-repo embedding sharing — currently each repo holds its own copy of the same embedding model.
- Per-language sub-function chunking that respects scope (today's chunker is line-based; smarter for very long classes).

## License

MIT
