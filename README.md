# DocGraph

A local code knowledge graph for LLMs. Indexes any repo with tree-sitter, stores entities and relationships in an embedded Kuzu file, and exposes everything via MCP and a live web graph UI. Runs **multiple repos at once** from a single host process — agents pick which repo per call via a closed-enum `root` argument.

```bash
pipx install docgraph
cd /any/repo
docgraph index                                       # ~5s for 100k LOC; sub-second incremental
docgraph host                                        # http://127.0.0.1:5500 — single repo (cwd)
docgraph host --root /repo-a --root /repo-b          # multi-root: one process, two repos
docgraph host --root /repo-a --watch /repo-a         # also reindex on change
docgraph mcp /repo-a --transport stdio               # editor stdio MCP (proxies through host if up)
```

### GUI / process supervision (optional)

Want a tray-based UI that supervises one `docgraph host` covering every repo you care about, and bridges every MCP tool into a local LLM through a proxy? See [telecode](https://github.com/prithwirajs/telecode) — its DocGraph section auto-starts the host, tails its log, and registers `docgraph_<tool>` in the managed-tools registry. The agent selects which repo per call via the `root` enum the host emits.

## Why

Most code-intelligence tools either ship a heavy multi-service stack (Neo4j + a vector DB + a separate UI app) or a thin keyword search. DocGraph keeps everything in one Python package backed by one file.

### Architecture

- **One Kuzu DB per repo, one host process per machine.** `docgraph host` runs the unified server: web UI + JSON API + MCP HTTP + optional watchers, all rooted in a `Workspace` registry that owns the per-repo connections. Multi-root via repeatable `--root` flags.
- **Closed-enum `root` selection.** The host reads its registered slugs at boot and emits the JSON API + MCP tool schemas with a JSON Schema enum. LLMs pick from a known set; protocol-layer rejection on typos. Single-root case collapses to a one-value default.
- **165+ languages** out of the box via tree-sitter (just install more `tree-sitter-*` packages).
- **Parallel indexer** — process pool, batched embeddings, bulk Cypher writes.
- **Per-file delta updates** — sub-second on edits, 0 ms on no-op runs.
- **Optional GPU acceleration** — `docgraph index --gpu` runs embeddings via ONNX Runtime on CUDA / DirectML / CoreML. Still no torch dep — just `pip install onnxruntime-gpu` / `onnxruntime-directml`. Falls back to CPU silently if no GPU runtime is installed.
- **Local-only by default** — no telemetry, no cloud round-trips. The only outbound network calls are opt-in: `docgraph docs add <url>` (you supply the URL) and `--llm-model <name>` (you supply the local server).

### Retrieval

- **MCP server** — 15 tools (6 base + 9 differentiators). Two transports: `stdio` for editors (Cursor / Claude Desktop) and `http` for web clients (`docgraph mcp --transport http`).
- **Differentiator edges** — `SIMILAR_TO` (vector top-K), `CO_CHANGED_WITH` (git history), `TESTS` (heuristic). The "what else will my change break?" answer.
- **Differentiator MCP tools** — `explore` (multi-hop BFS), `impact_of` (blast radius), `test_impact` (which tests cover this?), and `cypher` (raw read-only graph query — rejects `CREATE` / `MERGE` / `SET` / `DELETE` / `DROP` server-side, so agents can't accidentally mutate the graph).
- **Personalized PageRank** — `search` accepts `focus_file` / `focus_symbol` and ranks results by proximity to the file or symbol the agent is currently editing.
- **Cross-encoder reranker** — opt-in `search(rerank=True)` lifts top-K precision via a 33 MB Jina cross-encoder (still local, still ONNX).
- **Scope-aware resolution** — `CALLS` / `INSTANTIATES` / `INHERITS` prefer same-file then imported-file targets, killing most overload hallucinations without an LSP daemon.
- **Symbol-level imports + method overrides** — `IMPORTS_SYMBOL` (file → exact Class / Function it imports, not just file → file) and `OVERRIDES` (child method → parent method via the inheritance closure) so agents can ask "who imports this class?" and "what does this method override?" precisely.
- **Sub-function chunking** — long bodies split + embedded per chunk; search max-pools across chunks so a 1000-line class still has fine recall.
- **Diff- and history-aware tools** — `git_changes` (changed entities + 1-hop callers — Cursor `@Commit` joined to the graph), `git_blame` (line-range blame — Cursor `@Blame` parity), `git_recent` (last N commits scoped to a file or repo).

### Watcher + UI

- **Live graph UI** — single HTML file, no build step. Force-directed layout (ForceAtlas2-lite + label-propagation community detection) runs in a **Web Worker** so the main thread stays at 60fps. Render is Canvas 2D, batched by color and viewport-culled — comfortable up to ~10k nodes.
- **Detail Level / progressive reveal** — start with all `File` nodes (level 0); click any node to reveal its 1-hop neighbors. Lets you skim a 10k-node graph as a hub-and-spoke first, drill in only where you care.
- **Color modes** — color nodes by **kind** (Function / Class / File / …) or by **community** (auto-clustered via label propagation, no LLM).
- **Process detection** — entry-point → leaf call chains, surfaced in a dedicated **Processes** tab. An entry point = a function with no incoming `CALLS` edge; the panel shows ranked entries plus their forward call chain.
- **LLM-grounded wiki** — the **Wiki** tab generates one Markdown page per top-level module from a Kuzu fact sheet (top classes / functions by PageRank, importers, tests). CLI: `docgraph wiki`. Falls back to a plain rendering when the LLM is unreachable.
- **Watcher** — `docgraph watch` auto-reindexes on file changes (Rust `notify` under the hood, debounced).
- **Live UI auto-redraw** — `docgraph watch --serve` runs the watcher and the web server in **one process** so they share the Kuzu file lock. After every reindex the browser refreshes itself via Server-Sent Events at `/api/events` — no F5, no polling.
- **ML-training-style progress bars** — every index phase (parse, embed entities, embed chunks, write nodes, build symbol table, resolve edges, `SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`, PageRank, persist) reports `% | M/N | elapsed | ETA`. Same bars in `docgraph docs add`.

### Multi-root + ignores

- **Multi-root** — `docgraph host --root A --root B` runs one process serving N independent repos, each with its own `.docgraph/graph.kuzu`. Every API/MCP call accepts a `root=<slug>` arg (closed enum, validated at the protocol layer). The single web UI's repo picker is populated from `GET /api/roots`.
- **Indexer-side `--repo` (repeatable)** still works for monorepos: merges several path roots into a single index. Different from multi-root above (workspace registers independent indexes side-by-side; `--repo` builds one index from multiple paths).
- **Smart default ignores** — universal baseline (Cursor-parity: `node_modules/`, `__pycache__/`, `.venv/`, `.next/`, `.gradle/`, lockfiles, binaries, plus Jupyter / MLflow / wandb / DVC / R / Haskell / Zig caches) layered with per-ecosystem autodetect (Node / Python / Maven / Gradle / Rust / .NET / Angular / Android / Swift / Ruby / Dart / Elixir / Scala / PHP / Go / Terraform / Unity) — ambiguous build dirs (`target/`, `build/`, `bin/`, `obj/`) only ignored when their marker file is detected.
- **Two-tier ignore** — `.cursorindexingignore` skips files entirely; `.cursorignore` indexes them but redacts bodies/snippets returned to the AI. The HTTP API also sandboxes `/api/file_content` to the repo root (403 on traversal) and redacts `.cursorignore`'d files.
- **Cursor-rules compatible** — drops in existing `.cursor/rules/*.mdc` and `AGENTS.md`; exposes them via `rules_for(file)` so any MCP client gets glob-matched auto-attach.

### Optional augmentation

- **`@Docs` ingestion** — `docgraph docs add <url>` fetches and embeds external API docs; `search_docs(query)` MCP tool surfaces them. Idempotent (re-ingesting a URL replaces prior chunks).
- **Optional LLM-augmented docstrings** — opt-in via `--llm-model <name>`; talks to any OpenAI- or Anthropic-compatible local server (LM Studio, llama.cpp, vLLM, Ollama). DocGraph sends `reasoning_effort=none` so reasoning models (Qwen3, DeepSeek-R1) skip thinking and one-sentence summaries fit in a 150-token budget. Cached by body hash so incrementals stay fast.
- **LLM-grounded wiki (opt-in)** — `docgraph wiki` walks every top-level module, builds a fact sheet from Kuzu (top classes / functions / importers / tests), and asks the same local LLM to write a 200-300 word Markdown page per module. Saved to `.docgraph/wiki/<slug>.md` and shown in the Web UI's Wiki tab. Same `--llm-*` flags and `DOCGRAPH_LLM_*` env vars as `docgraph index`.

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

**All flags are optional.** Plain `docgraph index` with zero arguments works — flags only enable opt-in features (LLM docstrings, GPU embeddings, multi-repo) or alter defaults.

| Flag | Default | Description |
|---|---|---|
| `--full`, `-f` *(optional)* | `false` | Wipe the DB and rebuild from scratch instead of delta-update |
| `--repo PATH`, `-r PATH` *(optional)* | — | Additional repo root to include (repeatable). Persisted in `.docgraph/repos.json`; subsequent `watch` / `serve` / `mcp` pick it up automatically |
| `--llm-model STR` *(optional)* | unset (off) — when set, defaults to `qwen3.6-35b` via `$DOCGRAPH_LLM_MODEL` | **Activator.** Pass the model name your local server expects (e.g. `qwen3.6-35b`, `local-model`) to enable LLM-augmented docstrings for entities lacking native docs. Cached by body hash in `.docgraph/llm_docstrings.json` so incrementals don't re-call. |
| `--llm-port INT` *(optional)* | `1235` | Local LLM server port (host is always `localhost`). Ignored unless `--llm-model` is set. |
| `--llm-format STR` *(optional)* | `openai` | API format: `openai` (Chat Completions @ `/v1/chat/completions`) or `anthropic` (Messages @ `/v1/messages`). Ignored unless `--llm-model` is set. |
| `--llm-max-tokens INT` *(optional)* | `150` | Max tokens per LLM call. DocGraph sends `reasoning_effort=none` so reasoning models (Qwen3, DeepSeek-R1) fit a one-sentence answer in this budget; bump it for non-reasoning models if you want longer summaries. |
| `--gpu` *(optional)* | `false` | Use GPU for embeddings via ONNX Runtime (CUDA / DirectML / CoreML / ROCm). Requires `onnxruntime-gpu`, `onnxruntime-directml`, or `onnxruntime-silicon` to be installed. Falls back to CPU silently if no GPU runtime is found. |
| `--verbose`, `-v` *(optional)* | `false` | Verbose logs |

### `docgraph watch [path]`

Auto-reindex on file changes (Rust `notify` under the hood, debounced). Plain `watch` holds a writer lock — kill `serve` / `mcp` against the same DB first. With `--serve`, the watcher and the web UI run in **one process**, sharing the DB lock; the browser stays in sync via Server-Sent Events.

**All flags are optional.**

| Flag | Default | Description |
|---|---|---|
| `--debounce INT` *(optional)* | `500` | Debounce window in ms before reindex fires |
| `--serve` *(optional)* | `false` | Also run the web UI + JSON API in the same process. UI auto-redraws after each reindex via `/api/events` SSE. |
| `--host STR` *(optional)* | `127.0.0.1` | Bind address (only with `--serve`) |
| `--port INT` *(optional)* | `5500` | Bind port (only with `--serve`) |
| `--verbose`, `-v` *(optional)* | `false` | Verbose logs |

### `docgraph serve [path]`

Start the web UI + JSON API. **All flags are optional.**

| Flag | Default | Description |
|---|---|---|
| `--host STR` *(optional)* | `127.0.0.1` (or `$DOCGRAPH_HOST`) | Bind address |
| `--port INT` *(optional)* | `5500` (or `$DOCGRAPH_PORT`) | Bind port |
| `--verbose`, `-v` *(optional)* | `false` | Verbose access logs |

### `docgraph mcp [path]`

Run the Model Context Protocol server. **All flags are optional.**

| Flag | Default | Description |
|---|---|---|
| `--transport STR` *(optional)* | `stdio` | `stdio` (for Cursor / Claude Desktop) or `http` (for web clients) |
| `--verbose`, `-v` *(optional)* | `false` | Verbose logs |

### `docgraph stats [path]`

Print entity + edge counts. No flags.

### `docgraph wiki [path]`

Generate (or rebuild) an LLM-grounded wiki for the indexed repo. For every top-level module DocGraph pulls a fact sheet from Kuzu (top classes / functions by PageRank, importers, tests) and asks a local LLM to write a 200-300 word page. Pages land in `.docgraph/wiki/<slug>.md` and are surfaced in the Web UI's **Wiki** tab. If no LLM is reachable, the page falls back to a plain rendering of the facts so the wiki is never blank.

**Resumable.** If the run is interrupted (Ctrl-C, network blip, OOM), just run `docgraph wiki` again — modules whose page is already on disk are skipped without an LLM call. Pass `--force` to rebuild every page from scratch.

Uses the **same LLM config as `docgraph index --llm-model`**. All `DOCGRAPH_LLM_*` env vars are honored too.

**All flags are optional.** Like `docgraph index`'s LLM-docstring path, the only one you typically pass is `--llm-model` (most local servers reject unknown model names). Everything else has a working default.

| Flag | Default | Description |
|---|---|---|
| `--module STR`, `-m STR` *(optional)* | unset (all) | Build only the named top-level module. |
| `--llm-host STR` *(optional)* | `localhost` (or `$DOCGRAPH_LLM_HOST`) | Host running the local LLM server. |
| `--llm-port INT` *(optional)* | `1235` (or `$DOCGRAPH_LLM_PORT`) | Local LLM server port. |
| `--llm-model STR` *(optional)* | `qwen3.6-35b` (or `$DOCGRAPH_LLM_MODEL`) | Model name your local server expects. Override if your server uses a different identifier (`local-model`, `gpt-oss-20b`, etc.). |
| `--llm-format STR` *(optional)* | `openai` (or `$DOCGRAPH_LLM_FORMAT`) | API format: `openai` (Chat Completions) or `anthropic` (Messages). |
| `--llm-max-tokens INT` *(optional)* | `600` (or `$DOCGRAPH_LLM_MAX_TOKENS`) | Per-call token budget. Higher than `index`'s 150 because wiki pages are longer. |
| `--force`, `-f` *(optional)* | off | Rebuild every page from scratch. Default is resumable: skip modules whose page is already on disk. |

API equivalents (used by the Web UI's "Build wiki" button):

```
GET  /api/wiki/list                                # list of pages [{slug, title, module, summary}]
GET  /api/wiki/page?slug=<slug>                    # full Markdown body + facts JSON
POST /api/wiki/build  {"module": "X"?, "force": true?}  # rebuild all (or one module). Resumable by default; pass force=true to redo every page.
```

### `docgraph clear [path]`

Delete `.docgraph/` for the repo (DB + cache + repos list).

| Flag | Default | Description |
|---|---|---|
| `--yes`, `-y` *(optional)* | `false` | Skip the confirmation prompt |

### `docgraph daemon start`

Start the optional embedding daemon. Holds a single warm ONNX session in memory; other docgraph processes on this host route their embed calls through it via loopback TCP, cutting cold start to a TCP round trip. Foreground by default; pass `--detach` to background it. Lock file at `~/.docgraph/daemon.lock`.

**All flags are optional.**

| Flag | Default | Description |
|---|---|---|
| `--port INT` *(optional)* | `5577` | Loopback TCP port. 127.0.0.1 only — never exposed off-host. |
| `--model STR` *(optional)* | `BAAI/bge-small-en-v1.5` | Embedding model. Must match what your repos were indexed with, or vectors won't be comparable. |
| `--gpu` *(optional)* | `false` | Load the model on GPU via ONNX Runtime providers. Same opt-in install requirements as `docgraph index --gpu`. |
| `--detach`, `-d` *(optional)* | `false` | Spawn a background process and return. POSIX: double-fork; Windows: `DETACHED_PROCESS`. |
| `--verbose`, `-v` *(optional)* | `false` | Verbose logs |

Embedder integration is automatic: if the daemon is running when an `Embedder.embed()` call is made, the call gets routed through it. If the daemon is down or the protocol fails, the embedder loads its own session as before — never fails the request.

### `docgraph daemon stop`

Stop the running daemon. Idempotent — no-op if nothing's running. No flags.

### `docgraph daemon status`

Print whether the daemon is running and its config (pid, port, model, gpu flag, start time). Exits non-zero if no daemon is running. No flags.

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
| `~/.docgraph/daemon.lock` | (lock file, not env var) | Auto-managed by `docgraph daemon start` / `stop`. Contains `host`, `port`, `pid`, `model`, `gpu`, `started`. Other docgraph processes consult this to discover the running daemon. Stale locks are cleaned automatically. |
| `DOCGRAPH_LLM_MODEL` | `index`, `wiki` | unset for `index` (off — setting this enables LLM-augmented docstrings); `qwen3.6-35b` for `wiki`. |
| `DOCGRAPH_LLM_DOCSTRINGS` | `index` | unset. Set to `1`/`true` to enable explicitly (rarely needed; setting `DOCGRAPH_LLM_MODEL` is enough). |
| `DOCGRAPH_LLM_HOST` | `index`, `wiki` | `localhost` |
| `DOCGRAPH_LLM_PORT` | `index`, `wiki` | `1235` |
| `DOCGRAPH_LLM_FORMAT` | `index`, `wiki` | `openai` |
| `DOCGRAPH_LLM_API_KEY` | `index`, `wiki` | unset. If set, sent as `Authorization: Bearer …` (OpenAI) or `x-api-key: …` (Anthropic). |
| `DOCGRAPH_LLM_MAX_TOKENS` | `index`, `wiki` | `150` for `index` (one sentence); `600` for `wiki` (one page). DocGraph sends `reasoning_effort=none` so reasoning models (Qwen3 / DeepSeek-R1) skip thinking and fit in these budgets. |
| `DOCGRAPH_LLM_TIMEOUT` | `index`, `wiki` | `60` (seconds) |

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
| **Structural** | `CONTAINS`, `IMPORTS` (file → file or module), `IMPORTS_SYMBOL` (file → specific Class / Function imported by name, e.g. Python `from x import Y`, JS/TS `import {Y} from "x"`) |
| **Behavioral** | `CALLS`, `INSTANTIATES`, `REFERENCES_`, `RETURNS` |
| **Type system** | `INHERITS`, `IMPLEMENTS`, `OVERRIDES` (child→parent method via inheritance closure), `DECORATED_BY` |
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
| `GET /api/processes?limit=&max_chain_len=` | Detected entry-point → call chains. Used by the **Processes** tab. |
| `GET /api/wiki/list` | List of generated wiki pages (`{slug, title, module, summary}`). |
| `GET /api/wiki/page?slug=...` | Markdown body + facts JSON for one page. |
| `POST /api/wiki/build` (`{module?: "X", force?: true}`) | Rebuild all (or one module's) wiki page. Resumable: skips modules already on disk unless `force=true`. Uses the same LLM config as `docgraph index --llm-model` via `DOCGRAPH_LLM_*` env vars. |
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
pytest                   # ~65s (shared embedder cache + 178 tests, incl. daemon)
```

Covers indexer correctness, per-file delta updates, all retrieval methods, every MCP tool (registered + invoked end-to-end), every HTTP API route (incl. `.cursorignore` redaction + cypher write-blocker), multi-repo walking, watch filter logic, the embedding-text builder, and Variable-node round-trip + delete cascade.

Live LLM tests (`tests/test_llm_live.py`) auto-skip unless an OpenAI-compatible server is reachable at `localhost:1235` with `qwen3.6-35b` loaded. Override host/port/model via `DOCGRAPH_LLM_TEST_HOST` / `DOCGRAPH_LLM_TEST_PORT` / `DOCGRAPH_LLM_TEST_MODEL`.

## License

MIT
