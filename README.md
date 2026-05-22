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

Want a tray UI that supervises one `docgraph host` covering every repo you care about, and bridges every MCP tool into a local LLM via a proxy? See [telecode](https://github.com/prithwirajs/telecode) — its DocGraph section auto-starts the host, tails its log, and registers `docgraph_<tool>` in the managed-tools registry. The agent selects which repo per call via the `root` enum.

## Why

Most code-intelligence tools either ship a heavy multi-service stack (Neo4j + a vector DB + a separate UI app) or a thin keyword search. DocGraph keeps everything in one Python package backed by one file.

### Architecture

- **One Kuzu DB per repo, one host process per machine.** `docgraph host` runs the unified server: web UI + JSON API + MCP HTTP + optional watchers, all rooted in a `Workspace` registry that owns per-repo connections. Multi-root via repeatable `--root` flags.
- **Closed-enum `root` selection.** The host reads its registered slugs at boot and emits the JSON API + MCP tool schemas with a JSON Schema enum. LLMs pick from a known set; protocol rejects typos. Single-root collapses to a one-value default.
- **165+ languages** out of the box via tree-sitter (just `pip install` more `tree-sitter-*` packages).
- **Parallel indexer** — process pool, batched embeddings, bulk Cypher writes.
- **Per-file delta updates** — sub-second on edits, 0 ms on no-op runs.
- **Optional GPU acceleration** — `docgraph index --gpu` routes embeddings through torch via sentence-transformers. NVIDIA CUDA only; install with the matching torch wheel (`pip install --index-url https://download.pytorch.org/whl/cu130 torch`, or `cu124` / `cpu`). The CUDA wheels bundle their own CUDA + cuDNN runtime so no separate CUDA Toolkit install is needed. Falls back to CPU silently when no GPU is available, and falls back from CUDA to CPU mid-run on driver / OOM errors instead of crashing the host.
- **Local-only by default** — no telemetry, no cloud round-trips. The only outbound calls are opt-in LLM requests via `--llm-model <name>` (you supply the local server).
- **Configuration is flags-only.** No `DOCGRAPH_*` environment variables. Every knob is a CLI flag or a `load_config(...)` kwarg, so the spawn surface is fully visible in `ps` / Process Hacker.

### Retrieval

- **MCP server** — 15 tools (6 base + 9 differentiators). Two transports: `stdio` for editors (Cursor / Claude Desktop) and `http` for web clients (`docgraph mcp --transport http`).
- **Differentiator edges** — `SIMILAR_TO` (vector top-K), `CO_CHANGED_WITH` (git history), `TESTS` (heuristic). Answers "what else will my change break?".
- **Differentiator MCP tools** — `explore` (multi-hop BFS), `impact_of` (blast radius), `test_impact` (which tests cover this?), `cypher` (raw read-only graph query — rejects writes server-side).
- **Personalized PageRank** — `search` accepts `focus_file` / `focus_symbol` and ranks by proximity to what the agent is editing.
- **Cross-encoder reranker** — opt-in `search(rerank=True)` lifts top-K precision via a 33 MB Jina cross-encoder (local, torch).
- **Scope-aware resolution** — `CALLS` / `INSTANTIATES` / `INHERITS` prefer same-file then imported-file targets, killing most overload hallucinations without an LSP daemon.
- **Symbol-level imports + method overrides** — `IMPORTS_SYMBOL` (file → exact Class / Function imported by name) and `OVERRIDES` (child method → parent via the inheritance closure).
- **Sub-function chunking** — long bodies split + embedded per chunk; search max-pools across chunks so a 1000-line class still has fine recall.
- **Diff- and history-aware tools** — `git_changes` (changed entities + 1-hop callers), `git_blame` (line-range blame), `git_recent` (last N commits scoped to a file or repo).

### Watcher + UI

- **Live graph UI** — single HTML file, no build step. ForceAtlas2-lite + label-propagation community detection runs in a **Web Worker**; render is Canvas 2D, batched and viewport-culled — comfortable up to ~10k nodes.
- **Detail Level / progressive reveal** — start with all `File` nodes; click any node to reveal 1-hop neighbors. Skim a 10k-node graph as a hub-and-spoke first, drill in only where you care.
- **Color modes** — by **kind** (Function / Class / File / …) or by **community** (auto-clustered, no LLM).
- **Process detection** — entry-point → leaf call chains, surfaced in the **Processes** tab.
- **LLM-grounded wiki** — the **Wiki** tab generates one Markdown page per top-level module from a Kuzu fact sheet (top classes / functions by PageRank, importers, tests). CLI: `docgraph wiki`. Falls back to a plain rendering when the LLM is unreachable.
- **Watcher** — `docgraph host --watch <root>` auto-reindexes on file changes (Rust `notify`, debounced). The browser refreshes itself via SSE at `/api/events` — no F5, no polling.
- **Phase progress bars** — every index phase (parse, embed entities, embed chunks, write nodes, build symbol table, resolve edges, `SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`, PageRank, persist) reports `% | M/N | elapsed | ETA`.

### Multi-root + ignores

- **Multi-root** — `docgraph host --root A --root B` runs one process serving N independent repos, each with its own `.docgraph/graph.kuzu`. Every API/MCP call accepts a `root=<slug>` arg (closed enum, validated at the protocol layer). The single web UI's repo picker is populated from `GET /api/roots`.
- **Indexer-side `--repo` (repeatable)** still works for monorepos: merges several path roots into one index. Different from multi-root above.
- **Smart default ignores** — universal baseline (`node_modules/`, `__pycache__/`, `.venv/`, `.next/`, `.gradle/`, lockfiles, binaries, plus Jupyter / MLflow / wandb / DVC / R / Haskell / Zig caches, plus documentation-only files: `README*`, `CHANGELOG*`, `LICENSE*`, `CONTRIBUTING*`, `AUTHORS*`, `CODEOWNERS`) layered with per-ecosystem autodetect (Node / Python / Maven / Gradle / Rust / .NET / Angular / Android / Swift / Ruby / Dart / Elixir / Scala / PHP / Go / Terraform / Unity) — ambiguous build dirs only ignored when their marker file is detected.
- **Two-tier ignore** — `.cursorindexingignore` skips files entirely; `.cursorignore` indexes them but redacts bodies/snippets returned to the AI. The HTTP API also sandboxes `/api/file_content` to the repo root.
- **Cursor-rules compatible** — drops in existing `.cursor/rules/*.mdc` and `AGENTS.md`; exposes them via `rules_for(file)`.
- **External links** — each root can crawl external URLs alongside the code. Configure in `<root>/.docgraph/links.json`. BFS with `depth` (0 = seed page only; 1 = seed + all direct links), `max_pages` cap, and a `ttl_hours` staleness window. Fetched pages are indexed as `File` nodes with full embeddings and appear in search results alongside code. Re-fetched automatically when stale at the start of each index run.
- **Extra local paths** — `<root>/.docgraph/repos.json` lists sibling repo paths to fold into the same graph. Useful for monorepos where subdirectories live at different absolute paths.

### Optional augmentation

- **LLM-augmented docstrings (opt-in)** — `--llm-model <name>` enables it; talks to any OpenAI- or Anthropic-compatible local server (LM Studio, llama.cpp, vLLM, Ollama). DocGraph sends `reasoning_effort=none` so reasoning models (Qwen3, DeepSeek-R1) skip thinking and one-sentence summaries fit in a 150-token budget. Cached by body hash.
- **LLM-grounded wiki (opt-in)** — `docgraph wiki` walks every top-level module, builds a fact sheet from Kuzu, and asks the same local LLM to write a 200-300 word Markdown page per module. Saved to `.docgraph/wiki/<slug>.md` and shown in the Web UI.
- **Right-panel Chat tab (opt-in)** — when `--llm-model` is set, the Web UI's right panel adds a **Chat** tab next to **Detail**. It POSTs `/api/chat` against the same configured local LLM, renders Markdown + JSON in replies, and — when an entity is selected in the graph — automatically attaches that entity's snippet/file/language as a system-message preamble so the model has the source without any copy-paste. Chat output isn't capped on OpenAI-compatible servers (the model writes until done); the meta line tracks the active root and re-pulls config when you switch the root selector. The tab stays hidden when no LLM is configured.

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

Requires Python 3.10+. The first run downloads the embedding model (~130 MB BGE-small-en).

**Optional GPU acceleration** — install the torch wheel matching your CUDA version to enable `--gpu`. NVIDIA only; AMD / Intel GPU users stay on CPU.

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 torch   # NVIDIA + CUDA 13.x
pip install --index-url https://download.pytorch.org/whl/cu124 torch   # NVIDIA + CUDA 12.4
pip install --index-url https://download.pytorch.org/whl/cpu   torch   # CPU only (also default from PyPI)
```

The `+cuXY` wheels bundle their own CUDA + cuDNN runtime, so no separate CUDA Toolkit install is needed. DocGraph auto-detects with `torch.cuda.is_available()`; without a CUDA wheel it stays on CPU. CUDA OOM / driver errors mid-run are caught and the embedder falls back to CPU instead of crashing.

### Local dev install (Windows)

For working on docgraph itself rather than consuming it as a package:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
# .\setup.ps1 -Recreate                 # wipe .venv and reinstall
# .\setup.ps1 -CudaVersion cu130        # NVIDIA + CUDA 13.x (default)
# .\setup.ps1 -CudaVersion cu124        # NVIDIA + CUDA 12.4
# .\setup.ps1 -CudaVersion cpu          # CPU-only install
# .\setup.ps1 -NoShim                   # skip writing ~/.local/bin/docgraph.bat
```

Creates `.venv` next to the script, installs torch from PyTorch's per-CUDA index (whose `+cuXY` wheels bundle CUDA + cuDNN — no separate CUDA Toolkit install needed), runs `pip install -e .`, and drops a `docgraph.bat` shim into `~/.local/bin` so the CLI is on PATH. The repo's own `docgraph.bat` resolves the venv via `%~dp0` and works from any clone location.

## CLI reference

`path` argument defaults to the current directory; the repo root is auto-detected by walking up to find `.git`. **Every knob is a flag — there are no `DOCGRAPH_*` environment variables.**

### `docgraph index [path]`

Parallel index. Incremental by default; pass `--full` to wipe and rebuild.

| Flag | Default | Description |
|---|---|---|
| `--full`, `-f` | `false` | Wipe the DB and rebuild from scratch |
| `--repo PATH`, `-r PATH` | — | Additional repo root to fold into **the same** `.docgraph/graph.kuzu` (monorepo / sibling-projects shape). Persisted in `.docgraph/repos.json`. Different from host-side multi-root. |
| `--llm-model STR` | unset (off) | **Activator** for LLM-augmented docstrings. Pass the model name your local server expects (`qwen3.6-35b`, `local-model`, …). Cached by body hash. |
| `--llm-host STR` | `localhost` | Local LLM server host. Ignored unless `--llm-model` is set. |
| `--llm-port INT` | `1235` | Local LLM server port. Ignored unless `--llm-model` is set. |
| `--llm-format STR` | `openai` | API format: `openai` (Chat Completions) or `anthropic` (Messages). |
| `--llm-max-tokens INT` | `512` | Max tokens per LLM call. `reasoning_effort=none` lets reasoning models fit a one-sentence answer. |
| `--llm-prompt-docstring-file PATH` | unset | Custom docstring template (must keep `{kind}` / `{name}` / `{language}` / `{body}`). |
| `--gpu` | `false` | Use NVIDIA CUDA for embeddings via torch. Requires a `+cuXY` torch wheel installed (see Install). Falls back to CPU silently if `torch.cuda.is_available()` is False, and mid-run on CUDA OOM / driver errors. |
| `--workers INT` | `0` (auto) | Override worker count. `0` = `max(2, cpu_count - 1)`. |
| `--embed-batch-size INT` | `64` | Embedding batch size. Lower if you hit CUDA OOM with a larger model. |
| `--embed-model STR` | `BAAI/bge-small-en-v1.5` | Override the embedding model (any HF sentence-transformers id). Schema dim auto-aligns. Switching dim on an existing DB requires `clear` + reindex. |
| `--verbose`, `-v` | `false` | Verbose logs |

### `docgraph host [path]`

The unified server. One process serves N roots — web UI + JSON API + MCP HTTP all on the same port.

```bash
docgraph host                                       # cwd as the only root
docgraph host /repo-a                               # single-root sugar
docgraph host --root /repo-a --root /repo-b         # multi-root
docgraph host --root /repo-a --watch /repo-a        # also reindex on change
```

Accepts every `index`-time flag too (`--gpu`, `--embed-model`, `--llm-*`, `--rerank-default`, `--rerank-model`, `--rerank-gpu`) plus:

| Flag | Default | Description |
|---|---|---|
| `--root PATH`, `-r PATH` *(repeatable)* | — | Repo root to register. With multiple roots, every API/MCP call accepts `root=<slug>`. |
| `--watch PATH` *(repeatable)* | — | Per-root watcher. Each value must match a registered `--root`. |
| `--host STR` | `127.0.0.1` | Bind address |
| `--port INT` | `5500` | Bind port |
| `--debounce INT` | `500` | Watcher debounce (ms) |
| `--embed-idle-unload-sec FLOAT` | `0` | Unload the embedder after N idle seconds (0 = never). Reloads lazily. |
| `--rerank-idle-unload-sec FLOAT` | `0` | Unload the reranker after N idle seconds (0 = never). |
| `--embed-daemon` / `--no-embed-daemon` | off | Route embed + rerank to a shared daemon (see below). |
| `--daemon-port INT` | `5577` | Loopback port for the embedding daemon. |
| `--daemon-idle-exit-sec FLOAT` | `0` | Daemon exits after N idle seconds with both models unloaded, to free the CUDA context (0 = never). |
| `--llm-prompt-docstring-file PATH` | unset | Process-wide custom docstring template. |
| `--llm-prompt-wiki-file PATH` | unset | Process-wide custom wiki output-format tail (no placeholders required). |

### Embedding daemon (shared model, optional)

`docgraph daemon` is a loopback TCP server holding **one** warm embedder + cross-encoder reranker for the whole host. Other docgraph processes (the host, the watcher's reindex, CLI runs) route their embed/rerank calls through it, so there's a single model and a single ~300 MB CUDA context — requests queue through one session instead of each process loading its own copy.

```bash
docgraph daemon start --gpu --idle-exit-sec 600   # foreground; Ctrl+C to stop
docgraph daemon start -d --gpu                     # detached background
docgraph daemon status
docgraph daemon stop
```

Enable it for a host with `--embed-daemon` (the host spawns it lazily on first use). Two-stage idle management lives entirely in the daemon:

- `--embed-idle-unload-sec` / `--rerank-idle-unload-sec` — drop a model's **weights** after idle; reload lazily on the next request. No restart.
- `--idle-exit-sec` — once **both** models are unloaded and the daemon has been idle this long, it **exits** to release the CUDA context, and is respawned on the next embed/rerank. This is loop-safe: the daemon does no GPU work on boot and is only respawned on demand.

Without `--embed-daemon`, embedding/reranking happen in-process (pooled per host) with the same `*-idle-unload-sec` weight-unloading; the CUDA context then lives in the host until it exits.

### `docgraph watch [path]`

Auto-reindex on file changes. Now a thin alias for `docgraph host` with watchers — `docgraph watch <path>` is equivalent to a single-root host watching that path. `--serve` adds the web UI + JSON API + MCP HTTP.

### `docgraph serve [path]`

Thin alias for `docgraph host` with no watchers.

### `docgraph mcp [path]`

```bash
docgraph mcp /myrepo --transport stdio              # editor stdio MCP. Probes a running host first.
docgraph mcp /myrepo --transport stdio --standalone # explicit isolated mode (no host probe)
docgraph mcp /myrepo --transport http               # standalone HTTP MCP (prefer `docgraph host`)
```

| Flag | Default | Description |
|---|---|---|
| `--root PATH`, `-r PATH` *(repeatable)* | — | Repo root. Positional path is single-root sugar. |
| `--transport STR` | `stdio` | `stdio` (Cursor / Claude Desktop) or `http` |
| `--host STR` | `127.0.0.1` | Bind address (HTTP transport, or stdio's host probe) |
| `--port INT` | `5500` | Bind port (HTTP transport, or stdio's host probe) |
| `--host-url STR` *(stdio only)* | — | Override the URL stdio probes for an existing host |
| `--standalone` *(stdio only)* | `false` | Skip the host probe and run a single-process stdio server. |

**Strict-mode stdio.** With `--transport stdio` (default), `docgraph mcp <path>` first probes for a running `docgraph host`. If found, it acts as a thin proxy scoped to `<path>`. If `<path>` isn't a registered root on the host, it errors out — pass `--standalone` to bypass.

### `docgraph stats [path]`

Print entity + edge counts.

### `docgraph wiki [path]`

Generate (or rebuild) an LLM-grounded wiki. Resumable — re-running skips modules already on disk; `--force` rebuilds every page.

| Flag | Default | Description |
|---|---|---|
| `--module STR`, `-m` | unset (all) | Build only the named top-level module |
| `--llm-host STR` | `localhost` | LLM server host |
| `--llm-port INT` | `1235` | LLM server port |
| `--llm-model STR` | `qwen3.6-35b` | Model name your local server expects |
| `--llm-format STR` | `openai` | `openai` or `anthropic` |
| `--llm-max-tokens INT` | `4096` | Per-call token budget. Reasoning models still get `reasoning_effort=none`. |
| `--llm-prompt-wiki-file PATH` | unset | Custom wiki output-format tail |
| `--depth INT`, `-d` | `12` | Max directory levels to bucket files by. `1` = top-level only; `12` = one page per leaf folder. |
| `--force`, `-f` | off | Rebuild every page from scratch |

API equivalents:

```
GET  /api/wiki/list                                # [{slug, title, module, summary}]
GET  /api/wiki/page?slug=<slug>                    # full Markdown body + facts JSON
POST /api/wiki/build  {"module": "X"?, "force": true?}
```

### `docgraph clear [path]`

Delete `.docgraph/` for the repo (DB + cache + repos list).

| Flag | Default | Description |
|---|---|---|
| `--yes`, `-y` | `false` | Skip the confirmation prompt |

### `docgraph install-mcp [path]`

Print a JSON snippet ready to paste into Cursor / Claude Desktop's MCP config.

### `docgraph version`

Print version.

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
| `search(query, kind?, limit=10, focus_file?, focus_symbol?, rerank?)` | Hybrid vector + name + PageRank. `focus_*` → personalized PageRank; `rerank=True` → cross-encoder pass over the top candidates. |
| `definition(name, file?)` | Full body + metadata of a symbol |
| `references(name)` | All callers / usages |
| `call_graph(name, depth=2)` | Forward + backward call graph (depth 1–5) |
| `file_map(file)` | Entities + outgoing imports for a file |
| `neighborhood(name, limit=10)` | PageRank-ranked related code via calls + similarity + tests + inheritance |
| `explore(seeds, hops=3, limit=25)` | Multi-hop BFS subgraph from seed names |
| `impact_of(target, depth=3)` | Blast radius: transitive callers, importers, co-changed files, tests |
| `test_impact(target)` | Tests that exercise `target` via `TESTS` + reverse `CALLS*` |
| `cypher(query, limit=100)` | Read-only Cypher escape hatch. Rejects writes, caps rows |
| `git_changes(ref?)` | Diff-aware retrieval. `ref` = None / `HEAD` / `main` / `<sha>`. Returns files + entities + 1-hop callers |
| `git_blame(file, line_start, line_end?)` | `git blame` per line |
| `git_recent(file?, limit=20)` | Recent commits, optionally scoped to a file |
| `rules_for(file)` | Auto-attach rules: `.cursor/rules/*.mdc` glob match + `AGENTS.md` / `CLAUDE.md` always-on |
| `list_roots()` | `[{slug, path, default, watching, last_indexed_at}, …]` |

## Relationships extracted

| Tier | Edges |
|---|---|
| **Structural** | `CONTAINS`, `IMPORTS`, `IMPORTS_SYMBOL` (file → specific Class / Function imported by name) |
| **Behavioral** | `CALLS`, `INSTANTIATES`, `REFERENCES_`, `RETURNS` |
| **Type system** | `INHERITS`, `IMPLEMENTS`, `OVERRIDES` (child→parent method via the inheritance closure), `DECORATED_BY` |
| **Differentiators** | `SIMILAR_TO` (vector top-K), `CO_CHANGED_WITH` (git history), `TESTS` (heuristic name match) |

Nodes: `File`, `Module`, `Class`, `Function`, `Variable`, and `Chunk`. `Function` / `Class` carry an embedding + PageRank score; `Chunk` carries embeddings.

## Languages bundled

Out of the box: **python, javascript, typescript, tsx, java, go, rust, c, cpp, c_sharp, ruby, php, bash, html, css, json, yaml, markdown**.

Markdown files are indexed as sections keyed on ATX (`##`) and setext headings. Heading text is the entity name; the section body is embedded and searchable.

Adding more languages is two steps:

```bash
pip install tree-sitter-<lang>
```

then add an entry to `LANGUAGES` and a query to `TAGS_QUERIES` in `docgraph/parse.py`.

## Architecture

```
docgraph/
  cli.py             # typer entry: host (unified) / index / serve / mcp / watch / stats / wiki / clear
  workspace.py       # registry of registered roots — one host serves N roots, dynamic enum from slugs
  config.py          # load_config(repo_root, **overrides) — fully kwarg-driven; no env vars
  parse.py           # tree-sitter universal parser (per-language tags queries)
  index.py           # parallel pipeline + per-file delta updates
  db.py              # Kuzu schema + bulk insert (COPY FROM arrow)
  embed.py           # sentence-transformers (torch) wrapper + CUDA→CPU recovery
  rank.py            # PageRank over call + reference + inheritance graph
  retrieve.py        # hybrid retrieval (vector cosine + name boost + PageRank)
  rerank.py          # lazy Jina cross-encoder (~33 MB), GPU-capable
  llm.py             # urllib client + set_docstring_prompt(text) override
  mcp_tools.py       # 15 MCP tools + list_roots, all with closed-enum `root`
  mcp_stdio_proxy.py # strict stdio↔HTTP proxy for editors
  server.py          # FastAPI host: web UI + JSON API + SSE + FastMCP at /mcp
  watch.py           # per-root async awatch; one workspace-wide reindex semaphore
  wiki.py            # LLM-grounded module wiki + set_wiki_prompt_tail(text) override
  ui/index.html      # single-page force-directed canvas viewer (zero deps)
```

Data lives at `<repo>/.docgraph/`:
- `graph.kuzu/` — the embedded DB
- `cache.json` — per-file `{hash, entities, edges}` for delta updates
- `wiki/` — generated module pages
- `llm_docstrings.json` — body-hash-keyed cache of generated docstrings
- `repos.json` — extra sibling paths folded into this root's graph
- `links.json` — external URLs with crawl config `{url, depth, max_pages, ttl_hours}`

## How incremental works

1. Walk repo, SHA1-hash each file, compare to cache.
2. Bucket files into **changed / added / deleted / unchanged**.
3. `DETACH DELETE` only changed/deleted files' nodes — Kuzu drops incident edges in the same step.
4. Re-parse only changed/added files in a process pool.
5. Continue ID allocation from `max(id) + 1` in the DB.
6. Re-resolve only edges that touch a changed file; edges fully inside the unchanged set stay untouched.
7. Recompute `SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`, and PageRank — they're global and cheap.

## JSON API (when running `docgraph host`)

Every retriever route accepts a `root=<slug>` query parameter. The slug is one of those returned by `GET /api/roots`; on a single-root host it has one value and is the default.

| Endpoint | Notes |
|---|---|
| `GET /` | The web UI |
| `GET /api/roots` | `[{slug, path, default, watching, last_indexed_at}, …]` |
| `POST /api/admin/index` (`{full?: bool}`) | In-process incremental (or `full=true`) reindex via the workspace's writer-lock dance. Response: `{slug, full, stats, log}`. |
| `POST /api/admin/clear` | Wipe a root's index (DB + cache + wiki). Broadcasts a `reindex_done {events: -1}` SSE. |
| `POST /api/admin/cancel` | Cancel an in-flight `/api/admin/index` or `/api/wiki/build`. Returns 499 on the long-op. |
| `POST /mcp` | Mounted FastMCP HTTP transport |
| `GET /api/search?q=...&kind=...&limit=10` | Same as the MCP tool |
| `GET /api/definition`, `/references`, `/call_graph`, `/file_map`, `/neighborhood`, `/explore`, `/impact_of`, `/test_impact`, `/git_changes`, `/git_blame`, `/git_recent`, `/rules_for` | All MCP retriever tools as REST GETs |
| `POST /api/cypher` (`{query, limit}`) | Read-only Cypher |
| `GET /api/graph?limit_nodes=2000` | All nodes + edges for the viewer |
| `GET /api/stats` | Entity counts + per-edge-table counts |
| `GET /api/file_content?file=...` | Source text for inspection (sandboxed; redacts `.cursorignore`'d files) |
| `GET /api/processes?limit=&max_chain_len=` | Detected entry-point → call chains |
| `GET /api/wiki/list`, `?slug=`, `POST /api/wiki/build` | Wiki pages (resumable; `force=true` rebuilds) |
| `GET /api/llm_config` | Reports the active root's LLM augmentation knobs — `{configured, host, port, model, format, max_tokens, has_key}`. The web UI uses this to gate the right-panel **Chat** tab. |
| `POST /api/chat` (`{messages, context?, max_tokens?}`) | Multi-turn chat through the configured LLM. `messages` is an OpenAI-shaped `[{role, content}, …]` list. `context` (optional) is `{name, file, language, snippet}` and is injected as a system-message preamble so the model sees the entity's source. `max_tokens` is optional — omitted by default for OpenAI-compatible servers (model writes until done); Anthropic format forces a generous default since the API requires one. Returns `{content, model}`. |
| `GET /api/events` | SSE stream. Emits `reindex_done` after every reindex; the bundled UI uses it to auto-refresh. Keepalive every 15 s. |

## Comparison

| | DocGraph | GitNexus | Codebase-Memory | Cursor | Greptile | Sourcegraph (Cody) | Continue.dev |
|---|---|---|---|---|---|---|---|
| License | MIT | open | open | proprietary | proprietary SaaS | Apache 2 | Apache 2 |
| Runs fully local | ✅ | ✅ | ✅ | partial | ❌ (cloud) | ✅ (self-hosted) | ✅ |
| Embedded store | Kuzu (graph + vectors) | KuzuDB / LadybugDB | SQLite | proprietary | cloud | Postgres + cloud index | LanceDB / SQLite |
| Live graph UI | force-directed canvas + Web Worker physics | Mermaid (static) | ❌ | ❌ | ❌ | partial | ❌ |
| Per-file incremental | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Personalized PageRank | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cross-encoder rerank | ✅ (opt-in) | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `SIMILAR_TO` / `CO_CHANGED_WITH` / `TESTS` edges | ✅ | implicit/❌/❌ | ❌ | implicit/❌/❌ | implicit/❌/❌ | ❌ | implicit/❌/❌ |
| Diff-aware retrieval | ✅ (`git_changes`) | ❌ | ❌ | partial (`@Commit`) | ❌ | ❌ | ❌ |
| Cursor-rules ingest | ✅ (`.mdc` + `AGENTS.md`) | ❌ | ❌ | native | ❌ | ❌ | ❌ |
| Optional LLM docs | ✅ (local OpenAI/Anthropic-compat) | ❌ | ❌ | ❌ | ✅ (cloud) | ❌ | partial |
| Read-only Cypher escape hatch | ✅ | ❌ | ❌ | ❌ | ❌ | partial (GraphQL) | ❌ |
| MCP tools | 15 | 7 | 14 | n/a (IDE) | yes | via plugins | via plugins |
| Install | `pipx install` | manual | manual | proprietary IDE | hosted SaaS | self-host stack | binary |

## Multi-root

A single `docgraph host` process can serve any number of independently-indexed repos. Each registered root has its own `.docgraph/graph.kuzu`; the host opens a per-root read-only connection at startup, and every tool / route accepts a closed-enum `root=<slug>` argument.

```bash
docgraph index /path/to/repo-a
docgraph index /path/to/repo-b
docgraph host --root /path/to/repo-a --root /path/to/repo-b   # one process, one port
docgraph host --root /path/to/repo-a --watch /path/to/repo-a  # also reindex repo-a on file change
```

The workspace is **immutable for the host's lifetime** — adding/removing a root requires a host restart. Deliberate: keeps the closed-enum schema valid for the whole process. When supervised by telecode, the host restarts automatically when root paths are added or removed from the tray UI.

A different concept lives one layer down: `docgraph index --repo` lets the **indexer** walk multiple sibling paths into ONE `.docgraph/graph.kuzu` (useful for monorepos). The two shapes can coexist.

In multi-repo mode, file paths are prefixed with each repo's basename (`repo-b/src/foo.py`) so they stay unique.

## Tests

```bash
pip install pytest
pytest                   # ~90s, ~250 tests
```

Covers indexer correctness, per-file delta updates, all retrieval methods, every MCP tool (registered + invoked), every HTTP API route (incl. `.cursorignore` redaction + cypher write-blocker), multi-repo walking, watch filter logic, the embedding-text builder, Variable round-trip + delete cascade, the `Workspace` registry's `resolve()` + writer-lock round-trip, the GPU→CPU embedder fallback on a poisoned ORT session, every CLI flag telecode passes, the document/asset pass, and the env-free contract (`DOCGRAPH_*` env vars must not affect Config).

Live LLM tests (`tests/test_llm_live.py`) auto-skip unless an OpenAI-compatible server is reachable at `localhost:1235` with `qwen3.6-35b` loaded.

## License

MIT
