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

- **MCP server** — 26 tools (6 base + 9 differentiators + 11 analysis tools), MCP resources and prompts. Two transports: `stdio` for editors (Cursor / Claude Desktop) and `http` for web clients (`docgraph mcp --transport http`).
- **Call-resolution confidence** — every `CALLS` / `INSTANTIATES` / `INHERITS` edge carries `confidence` + `method` from a cascade: import map 0.95, same module 0.90, imported file 0.85, unique name 0.75, import distance 0.55, fuzzy 0.3–0.4 (SCIP: 1.0). Receiver-aware: `re.search()` never resolves to your `search`, `obj.m()` only to methods, a bare `f()` never to another file's method. Ambiguous call sites are kept as `CALLS_CANDIDATE` rows instead of invented edges. `min_confidence` on `call_graph` / `impact_of` / `explore` / `test_impact` / `trace` / `processes` and in the UI.
- **One-call context** — `context(symbol, tokens=2000)`: definition, signature/doc, callers and callees ranked by PageRank x confidence, tests, flows (routes / MCP tools / entry points) it takes part in, cluster, recent commits, rules, similar symbols, trimmed to a token budget.
- **Change impact** — `detect_changes(ref | diff)`: changed symbols -> transitive callers, affected flows, tests to run (with a `pytest` command) and an explainable 0–100 risk score per symbol and overall.
- **Repo map** — `repo_map(focus=[files/symbols], tokens=1024)`: Aider-style signatures-only map ranked by personalized PageRank, binary-searched to fit the budget.
- **Communities** — Louvain clusters computed at index time and stored in the graph (`Community` + `MEMBER_OF`), auto-named, with cohesion; `list_clusters` / `cluster`.
- **Framework maps** — HTTP routes (FastAPI, Flask, aiohttp, Starlette, Express) and MCP tools / resources / prompts become `Route` / `Tool` nodes with `HANDLES` edges; flows start from them; `route_map`, `api_impact(route)`.
- **`trace(a, b)`**, **`health`** (hubs, bridges, dead code, import cycles, large functions, untested hotspots), **`symbol_history`** (first seen / last changed commit per symbol), **`rename`** (graph + text edit plan; writes only through the API with an explicit flag).
- **Differentiator edges** — `SIMILAR_TO` (vector top-K), `CO_CHANGED_WITH` (git history), `TESTS` (heuristic). Answers "what else will my change break?".
- **Differentiator MCP tools** — `explore` (multi-hop BFS), `impact_of` (blast radius), `test_impact` (which tests cover this?), `cypher` (raw read-only graph query — rejects writes server-side).
- **Personalized PageRank** — `search` accepts `focus_file` / `focus_symbol` and ranks by proximity to what the agent is editing.
- **Cross-encoder reranker** — opt-in `search(rerank=True)` lifts top-K precision via a 33 MB Jina cross-encoder (local, torch).
- **Scope-aware resolution** — `CALLS` / `INSTANTIATES` / `INHERITS` prefer same-file then imported-file targets, killing most overload hallucinations without an LSP daemon.
- **Symbol-level imports + method overrides** — `IMPORTS_SYMBOL` (file → exact Class / Function imported by name) and `OVERRIDES` (child method → parent via the inheritance closure).
- **Sub-function chunking** — long bodies split + embedded per chunk; search max-pools across chunks so a 1000-line class still has fine recall.
- **Diff- and history-aware tools** — `git_changes` (changed entities + 1-hop callers), `git_blame` (line-range blame), `git_recent` (last N commits scoped to a file or repo).

### Watcher + UI

- **Web UI** — one self-contained HTML file (`docgraph/ui/index.html`), no build step, no CDN. Eight pages on hash routes, a root picker, a `Ctrl K` / `/` jump palette, light/dark, and a layout that works on desktop (three panes with drag handles), tablet (detail drawer) and phone (bottom tab bar, explorer drawer, detail bottom sheet):
  - **Graph** (`#/graph`) — ForceAtlas2-lite layout + label-propagation communities in a **Web Worker**; Canvas 2D render batched by colour, viewport-culled, labels capped by PageRank — comfortable up to ~10k nodes. Explore (filter loaded nodes, Enter = semantic search), Filters (node/edge types, focus depth, colour by kind or community, level of detail), Files. The detail pane has Overview (definition, AI summary, references, related, community, matching rules), Code (source with Preview for Markdown/HTML), Calls (call graph, depth 1–3), Impact (`impact_of` + `test_impact`, copyable pytest command) and History (`git_recent` + `git_blame`).
  - **Search** (`#/search`) — hybrid search with kind chips, optional rerank and "near the selection" boost, with a source preview; or a read-only **Cypher** console.
  - **Wiki** (`#/wiki`) — module pages with a table of contents and sources; build missing pages, rebuild all, or rebuild one page, with live progress.
  - **Flows** (`#/flows`) — flows starting at routes, MCP tools and entry points, each drawable as an inline-SVG **sequence diagram**; a **Routes & tools** tab lists every detected route / tool with what it reaches (`api_impact`).
  - **Changes** (`#/changes`) — `detect_changes` for the working tree, the last commit, `main`, any ref, or a pasted diff: risk score + badges with the factor explanation, changed symbols, callers, affected flows, tests to run (copyable command), the diff, and **Show on graph** (diff overlay: changed nodes and affected callers highlighted).
  - **Insights** (`#/insights`) — `health` (hubs, bridges, dead code, import cycles, large functions, untested hotspots), the cluster table and detail, a **Trace** tool and a **Repo map** tool.
  - The graph can colour by **server-side cluster** with **named hulls**, and filter CALLS edges by **min confidence**. The detail pane adds **Context** (token-budget control), **History** (introduced / last changed + `git log -L`) and **Rename** (dry-run plan; applying needs a typed confirmation).
  - **Ask** (`#/ask`) — chat with the configured LLM, with a symbol's source attached as context; threads are kept in the viewer's browser.
  - **Index** (`#/index`) — roots with counts and status, index/wiki/fetch jobs with live phase progress and cancel, model load state, sibling repos and external links, locks, a Cypher console, and a confirmed "clear index".
- **Detail Level / progressive reveal** — start with all `File` nodes; click any node to reveal 1-hop neighbors. Skim a 10k-node graph as a hub-and-spoke first, drill in only where you care.
- **Color modes** — by **kind** (Function / Class / File / …) or by **community** (auto-clustered, no LLM).
- **Process detection** — entry-point → leaf call chains, surfaced on the **Flows** page.
- **LLM-grounded wiki** — the **Wiki** page generates one Markdown page per top-level module from a Kuzu fact sheet (top classes / functions by PageRank, importers, tests). CLI: `docgraph wiki`. Falls back to a plain rendering when the LLM is unreachable.
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
- **Ask page (chat)** — the Web UI's **Ask** page POSTs `/api/chat` against the same configured local LLM and renders Markdown + JSON in replies. **Ask about this** on a graph node (or **+ Context** on the page) attaches that entity's snippet/file/language as a system-message preamble, so the model has the source without any copy-paste; inline `code` that names an indexed symbol links back to the graph. Chat output isn't capped on OpenAI-compatible servers (the model writes until done); the meta line shows the model and the active root. Threads live in the viewer's browser (`localStorage`), not on the host.

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
| `--embed-model STR` | `BAAI/bge-small-en-v1.5` | Override the embedding model (any HF sentence-transformers id). Schema dim auto-aligns. Switching dim on an existing DB requires `clear` + reindex. See **Embedding models** below. |
| `--embed-cache / --no-embed-cache` | on | Reuse vectors of entities whose content hash is unchanged (moves / renames / re-parses) on incremental runs. |
| `--history / --no-history` | on | Record `first_seen` / `last_changed` commit per symbol (one `git blame` per changed file). |
| `--history-max-files INT` | `5000` | Cap on files blamed per run (`0` disables history). |
| `--communities / --no-communities` | on | Detect communities (Louvain) at index time. |
| `--scip STR` | `auto` | Precise SCIP references: `auto` (use `scip-python` / `scip-typescript` from PATH or `--scip-index` when present), `on` (always re-run the binaries), `off`. |
| `--scip-python PATH` / `--scip-typescript PATH` | PATH lookup | Explicit SCIP indexer binaries. |
| `--scip-index PATH` | unset | A prebuilt `index.scip` to ingest instead of running a binary. |
| `--verbose`, `-v` | `false` | Verbose logs |

### `docgraph host [path]`

The unified server. One process serves N roots — web UI + JSON API + MCP HTTP all on the same port.

```bash
docgraph host                                       # cwd as the only root
docgraph host /repo-a                               # single-root sugar
docgraph host --root /repo-a --root /repo-b         # multi-root
docgraph host --root /repo-a --watch /repo-a        # also reindex on change
```

Accepts every `index`-time flag too (`--gpu`, `--embed-model`, `--llm-*`, `--rerank-default`, `--rerank-model`, `--rerank-gpu`, `--embed-cache`, `--history`, `--history-max-files`, `--communities`, `--scip*`; they apply to index runs started from the host) plus:

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

### Embedding models

Any sentence-transformers id works with `--embed-model`. Code-specific models get a built-in profile (`docgraph embed-models`) so nothing else needs configuring: the vector size is derived for the Kuzu schema, query / document prefixes are applied where the model needs them (queries only, via `Embedder.embed_query`), `trust_remote_code` is switched on only for models whose architecture is Hub code, and long-context models are capped at 1024 tokens per input.

| Model | Dim | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (default) | 384 | general text, fast on CPU |
| `nomic-ai/CodeRankEmbed` | 768 | code retriever; query prefix; remote code; needs `pip install einops` |
| `Qodo/Qodo-Embed-1-1.5B` | 1536 | 1.5B code embedder; GPU strongly recommended (6.2 GB download) |
| `jinaai/jina-embeddings-v2-base-code` | 768 | profile present, but its Hub remote code does not load on transformers >= 5 |
| `Salesforce/SFR-Embedding-Code-400M_R`, `nomic-ai/nomic-embed-text-v1.5`, `intfloat/e5-*-v2` | 1024 / 768 / 384-1024 | prefixes / remote code handled |

**Switching model changes the vector size: `docgraph clear` (or `POST /api/admin/clear`) and a full reindex are required.**

Measured on this repository (65 files, 1,135 entities, full index on an RTX 5070 Ti; `docgraph bench --runner search`, 15 questions, recall@5 / MRR of the reference symbols through the full hybrid search):

| Model | recall@5 | MRR | embed time (GPU) |
|---|---|---|---|
| bge-small-en-v1.5 | 0.533 | 0.462 | ~5 s |
| CodeRankEmbed | 0.467 | 0.325 | ~25 s |
| Qodo-Embed-1-1.5B | 0.633 | 0.519 | 68 s |

The default stays `bge-small-en-v1.5`: CodeRankEmbed is worse here, and Qodo's gain (1–2 of 15 questions) does not justify a 6 GB model, a GPU requirement and 4x larger vectors as a default. Pick Qodo explicitly if you have the GPU.

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

### `docgraph agent-setup [path]`

Write DocGraph skill files and hooks for Claude Code, Codex and agy into a repo. Runs only when you invoke it, is idempotent (a second run reports every file `unchanged`), and previews with `--dry-run`.

| Writes | For |
|---|---|
| `.claude/skills/docgraph/SKILL.md`, `.claude/skills/docgraph-area-<cluster>/SKILL.md` | Claude Code: how to use the tools + one skill per detected cluster |
| `.claude/settings.json` (merged) | Claude Code `PreToolUse` hook on `Edit|Write|MultiEdit` -> `docgraph hook pre-edit` (impact hint as `additionalContext`, never blocks) |
| `AGENTS.md` (a marked block), `.agents/skills/docgraph*/SKILL.md` | Codex and agy |
| `.git/hooks/post-commit` (a marked block) | `docgraph hook post-commit`: "the index is now stale" notice |

| Flag | Default | Description |
|---|---|---|
| `--target`, `-t` | all | `claude`, `codex`, `agy` or `all` (repeatable) |
| `--host-url` | `http://127.0.0.1:5500` | Host the skills and hooks point at |
| `--root` | — | Root slug on that host |
| `--hooks / --no-hooks` | on | Write the Claude hook and the git post-commit hook |
| `--clusters / --no-clusters` | on | One skill per cluster (`--max-clusters`, default 12) |
| `--docgraph-cmd` | `docgraph` | Command the hooks run |
| `--dry-run` | off | Show the plan, write nothing |

Existing content is preserved: JSON is merged (only DocGraph's hook entry is replaced), Markdown lives between `<!-- docgraph:begin -->` / `<!-- docgraph:end -->`, git hooks get a marked block.

### `docgraph hook pre-edit | post-commit`

The entry points the hooks above call (`--host-url`, `--root`). `pre-edit` reads the Claude hook JSON on stdin; both always exit 0.

### `docgraph embed-models`

List the embedding models with built-in profiles (dimension, query prefix, remote code).

### `docgraph bench [path]`

Benchmark answers to a question set (`docgraph/bench_questions.json`: 15 questions about this repo across where-defined, callers, impact, flow, architecture and tests-to-run, each with a reference answer and keywords).

| Runner (`--runner`, repeatable) | What it does |
|---|---|
| `docgraph` | Offline: `search` + `context` through the host REST API (`--host-url`, `--root`) |
| `grep` | Offline baseline: keyword grep of the repo |
| `search` | Offline, in-process retrieval quality: recall@5 / MRR of the reference symbols (`--embed-model`, `--gpu`) — used to compare embedding models |
| `telecode-mcp` / `telecode-grep` | A real agent through telecode's Task API (`--telecode-url`, default `http://127.0.0.1:1235`): `CLAUDE_CODE`, `--model haiku`, `is_local: false` (cloud only). `mcp` tells the agent to use DocGraph's REST tools; `grep` to use grep / file reads only. `--judge` adds an LLM grade (0–10) against the reference. |

Scores: keyword recall, judge score, tokens and tool calls per answer (`--ids`, `--limit`, `--out report.json`).

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
| `context(symbol, file?, tokens=2000, min_confidence=0)` | 360 view trimmed to a token budget: definition + signature/doc, callers/callees ranked by PageRank x confidence, tests, flows, cluster, recent commits, rules, similar symbols; `text` is ready-to-read Markdown |
| `detect_changes(ref?, diff?, depth=3, min_confidence=0)` | Diff (git ref or unified diff text) -> changed symbols, callers, affected flows, tests to run + `test_command`, explainable risk score, graph overlay ids |
| `repo_map(focus?, tokens=1024, exclude_tests=True)` | Aider-style signatures-only map, personalized PageRank to `focus`, fitted to the budget |
| `list_clusters(limit)` / `cluster(id? name?)` | Communities: name, size, cohesion, top members, files, inter-cluster links / members, API, depends-on, used-by |
| `route_map(filter?)` / `api_impact(route, depth=4)` | HTTP routes + MCP tools with handlers and reach / what one route touches (functions, files, tests, routes sharing dependencies) |
| `trace(a, b, max_depth=8, min_confidence=0)` | Shortest directed path over CALLS + member edges; tries b -> a if needed |
| `health(limit=15, min_confidence=0.5)` | Hubs, bridges (approx. betweenness), dead code, import cycles, large functions, untested hotspots |
| `symbol_history(name, file?)` | First seen / last changed commit, `git log -L` history, removed symbols |
| `rename(symbol, new_name, file?, include_text=True)` | Edit plan only (graph-confirmed + text matches, before/after per line); applying is only possible via `POST /api/rename` |

`call_graph`, `impact_of`, `explore` accept `min_confidence` too.

**MCP resources:** `docgraph://schema`, `docgraph://clusters`, `docgraph://cluster/{id}`, `docgraph://flows`, `docgraph://flow/{id}`, `docgraph://routes` (default root) and `docgraph://roots/{slug}/clusters|flows|routes`. **MCP prompts:** `detect_impact(ref?, root?)` (pre-commit review with risk, tests to run) and `architecture_map(root?)` (cluster map as Mermaid).

## Relationships extracted

| Tier | Edges |
|---|---|
| **Structural** | `CONTAINS`, `IMPORTS`, `IMPORTS_SYMBOL` (file → specific Class / Function imported by name) |
| **Behavioral** | `CALLS {line, confidence, method}`, `CALLS_CANDIDATE` (the candidates of an ambiguous call site), `INSTANTIATES {confidence, method}`, `REFERENCES_`, `RETURNS` |
| **Type system** | `INHERITS {confidence, method}`, `IMPLEMENTS`, `OVERRIDES` (child→parent method via the inheritance closure), `DECORATED_BY` |
| **Differentiators** | `SIMILAR_TO` (vector top-K), `CO_CHANGED_WITH` (git history), `TESTS` (heuristic name match) |
| **Architecture** | `MEMBER_OF` (File / Class / Function → `Community`), `HANDLES` (`Route` / `Tool` → handler Function) |

Nodes: `File`, `Module`, `Class`, `Function`, `Variable`, `Chunk`, `Community`, `Route`, `Tool`. `Function` / `Class` carry an embedding, PageRank, a content hash (`ehash`, the embedding-cache key) and `first_seen_commit` / `last_changed_commit`; `Chunk` carries embeddings.

**Optional precise references (SCIP).** If `scip-python` / `scip-typescript` are on PATH (or passed with `--scip-python` / `--scip-typescript`), or a prebuilt index is given with `--scip-index`, the indexer ingests the SCIP occurrences (a reference inside function A to a symbol defined by function B) as `CALLS` with confidence 1.0 / method `scip`, upgrading heuristic edges in place. The protobuf is read by a small built-in decoder (no protobuf dependency). Without a binary or index you get one status line (`SCIP: ... skipped`) and `state.json["scip"]`.

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
  retrieve.py        # hybrid retrieval (vector cosine + name boost + PageRank) + analysis tools
  resolve.py         # call-resolution confidence cascade
  frameworks.py      # HTTP route + MCP tool detection (table-driven)
  communities.py     # Louvain communities
  insights.py        # diff parsing, risk scoring, token budgets, repo-map rendering
  merkle.py          # file-hash tree for fast scans
  history.py         # git blame symbol history
  scip.py            # optional SCIP ingest (built-in protobuf decoder)
  agent_setup.py     # `docgraph agent-setup` + hook runtime
  bench.py           # `docgraph bench` harness (+ bench_questions.json)
  rename.py          # apply a rename plan (API only)
  rerank.py          # lazy Jina cross-encoder (~33 MB), GPU-capable
  llm.py             # urllib client + set_docstring_prompt(text) override
  mcp_tools.py       # 26 MCP tools + resources + prompts, all with closed-enum `root`
  mcp_stdio_proxy.py # strict stdio↔HTTP proxy for editors
  server.py          # FastAPI host: web UI + JSON API + SSE + FastMCP at /mcp
  watch.py           # per-root async awatch; one workspace-wide reindex semaphore
  wiki.py            # LLM-grounded module wiki + set_wiki_prompt_tail(text) override
  ui/index.html      # the whole web UI: 8 hash-routed pages + Web Worker graph (zero deps)
```

Data lives at `<repo>/.docgraph/`:
- `graph.kuzu/` — the embedded DB
- `cache.json` — per-file `{hash, entities, edges}` for delta updates
- `merkle.json` — per-file `(size, mtime, sha1)` + directory hashes (the file-hash tree; a cache, safe to delete)
- `state.json` — schema version, git heads, resolution stats, embedding-cache hits, SCIP status, history bookkeeping, removed symbols
- `wiki/` — generated module pages
- `llm_docstrings.json` — body-hash-keyed cache of generated docstrings
- `repos.json` — extra sibling paths folded into this root's graph
- `links.json` — external URLs with crawl config `{url, depth, max_pages, ttl_hours}`

## How incremental works

0. If `state.json["schema_version"]` differs from the code's schema version, run a full reindex once (**upgrading to schema v3 triggers this automatically on the next index run**).
1. Walk repo; hash files through the file-hash tree (`.docgraph/merkle.json`): a file whose size + mtime match the last scan reuses its hash without being read; directory hashes form a Merkle tree so unchanged subtrees are known.
2. Bucket files into **changed / added / deleted / unchanged**.
3. Harvest the embeddings of the nodes about to be deleted (keyed by content hash), then `DETACH DELETE` only changed/deleted files' nodes — Kuzu drops incident edges in the same step.
4. Re-parse only changed/added files in a process pool; `git blame` them for symbol history.
5. Continue ID allocation from `max(id) + 1` in the DB. Entities whose content hash was harvested (moved / renamed / re-parsed but identical) reuse their vector instead of being re-embedded.
6. Re-resolve only edges that touch a changed file through the confidence cascade; edges fully inside the unchanged set stay untouched (so a new same-named symbol elsewhere does not retro-actively re-resolve an unchanged caller until a full reindex).
7. Recompute `SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`, PageRank and communities — they're global and cheap.

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
| `GET /api/llm_config` | Reports the active root's LLM augmentation knobs — `{configured, host, port, model, format, max_tokens, has_key}`. The web UI's Ask page and Index page show it. |
| `POST /api/chat` (`{messages, context?, max_tokens?}`) | Multi-turn chat through the configured LLM. `messages` is an OpenAI-shaped `[{role, content}, …]` list. `context` (optional) is `{name, file, language, snippet}` and is injected as a system-message preamble so the model sees the entity's source. `max_tokens` is optional — omitted by default for OpenAI-compatible servers (model writes until done); Anthropic format forces a generous default since the API requires one. Returns `{content, model}`. |
| `GET /api/events` | SSE stream. Emits `reindex_done` after every reindex (the bundled UI reloads the graph), plus `index_progress` / `wiki_progress` `{job_id, repo_slug, phase, current, total}` during jobs (the UI's progress bars). Keepalive every 15 s. |
| `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel` | Index / wiki / fetch jobs of this host process (`?root=` filters by repo **path**). In-memory, so they skip the read gate. |
| `GET/POST/DELETE /api/repos`, `GET/POST/DELETE /api/links`, `POST /api/links/fetch` | Sibling repo paths and external links of a root (the Index page's Sources card). |
| `GET /api/locks`, `GET /api/admin/models_status` | Writer-lock state per root and pooled embedder/reranker load state. |
| `GET /api/context?symbol=&file=&tokens=&min_confidence=` | `context` tool |
| `GET /api/detect_changes?ref=&depth=&min_confidence=`, `POST /api/detect_changes {diff, ref?, depth?, min_confidence?}` | `detect_changes` for a git ref or a pasted unified diff |
| `GET /api/repo_map?focus=a,b&tokens=&exclude_tests=` | `repo_map` (comma-separated focus) |
| `GET /api/clusters`, `GET /api/cluster?id=|name=` | Communities (404 for an unknown id) |
| `GET /api/routes?filter=`, `GET /api/api_impact?route=` | Route / tool map and one route's impact |
| `GET /api/trace?a=&b=&max_depth=&min_confidence=` | Shortest call path |
| `GET /api/health?limit=&min_confidence=` | Health report |
| `GET /api/symbol_history?name=&file=` | Symbol history |
| `GET /api/flow?id=` | One flow by entry function id (`processes` rows now carry `id`, `kind`, `via`, `edges`) |
| `POST /api/rename {symbol, new_name, file?, include_text?, dry_run?, apply?, sources?}` | Returns the plan. Writes **only** with `dry_run: false` **and** `apply: true` (default sources `["graph"]`); every edit is re-verified against the file |
| `GET /api/index_info` | Schema version / reindex-required, resolution stats per tier, embedding-cache hits, scan stats, communities, SCIP status, capabilities |

`/api/call_graph`, `/api/impact_of`, `/api/explore`, `/api/test_impact`, `/api/processes` take `min_confidence`; `/api/graph`, `/api/files`, `/api/node_neighbors` nodes carry `cluster` and CALLS edges `confidence`.

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
| MCP tools | 26 (+ resources, prompts) | 7 | 14 | n/a (IDE) | yes | via plugins | via plugins |
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
pytest -p no:cacheprovider   # ~310 tests
```

Covers indexer correctness, per-file delta updates, all retrieval methods, every MCP tool (registered + invoked), every HTTP API route (incl. `.cursorignore` redaction + cypher write-blocker), multi-repo walking, watch filter logic, the embedding-text builder, Variable round-trip + delete cascade, the `Workspace` registry's `resolve()` + writer-lock round-trip, the GPU→CPU embedder fallback on a poisoned ORT session, every CLI flag telecode passes, the document/asset pass, and the env-free contract (`DOCGRAPH_*` env vars must not affect Config). `tests/test_graph_features.py` covers schema v3 on a second fixture repo (`tests/fw_fixture.py`: FastAPI / Flask / aiohttp / Express routes, MCP tools, an ambiguous call, an external receiver, an import cycle, dead code, two commits and a synthetic SCIP index): confidence tiers, candidates, communities, routes, context, detect_changes, repo_map, trace, health, history, rename (plan + line-verified apply), MCP resources/prompts, every new REST route, and the embedding cache + file-hash tree + removed-symbol tracking across incremental runs.

Live LLM tests (`tests/test_llm_live.py`) auto-skip unless an OpenAI-compatible server is reachable at `localhost:1235` with `qwen3.6-35b` loaded.

## License

MIT
