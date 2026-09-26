# DocGraph — Working Notes for Claude

Things you can't infer from the code in 10 seconds. Read before non-trivial changes.

## What this is

Local code knowledge graph. Tree-sitter parses, sentence-transformers (torch) embeds, Kuzu stores, FastAPI + FastMCP serve. One `docgraph host` process serves **multiple roots** — one uvicorn, one MCP endpoint, one log, with a closed-enum `root` arg on every tool/route. (v2 rewrite of an old Neo4j + Chroma + Streamlit stack, preserved at tag `v1-legacy` — don't resurrect.)

## Hard rules

- **One package, one process, one DB file per root.** No new top-level dirs, no separate frontend builds, no microservices.
- **Workspace is immutable for the host's lifetime.** Adding/removing roots needs a restart (telecode does this). No hot-reload endpoints.
- **Kuzu is the only store.** No SQLite/Chroma/Neo4j/Redis.
- **No npm.** UI is one HTML file at `docgraph/ui/index.html`.
- **Embeddings via torch + sentence-transformers.** GPU is opt-in (`--gpu`); silently falls back to CPU if `torch.cuda.is_available()` is False. Install the matching torch wheel from `download.pytorch.org/whl/cuXY`; default PyPI wheel is CPU-only.
- **No env vars.** All config is `load_config(...)` kwargs / CLI flags. Zero `DOCGRAPH_*` reads.
- **Per-language processor classes are forbidden.** Add a language = two dict entries in `parse.py` (`LANGUAGES` + `TAGS_QUERIES`).
- **Tools/routes use a closed-enum `root`** built from workspace slugs at boot. Single root → one defaulted value (LLM doesn't see it).

## File map

| File | Purpose |
|---|---|
| `cli.py` | Typer entry. `host` is the unified command; `serve`/`mcp`/`watch` delegate to it. `daemon` subcommands manage the shared model daemon. Every config knob is a flag. |
| `config.py` | `load_config(repo_root, **overrides)` — kwarg-driven Config. Auto-detects ecosystems, respects `.gitignore`/`.docgraphignore`. `extra_roots` indexes multiple paths into ONE DB. r/w `<root>/.docgraph/{repos,links}.json`. |
| `index.py` | Parallel pipeline + per-file delta. **Most complex file.** Cache/state writes are atomic (tmp + `os.replace`). |
| `db.py` | Kuzu schema + bulk insert. Edges via `COPY FROM arrow`. `close()` is explicit (Windows + COPY won't release the lock on `del`). |
| `workspace.py` | `Workspace` registry of `RootSlot`s. Each owns an RO Kuzu conn + Retriever + pooled Embedder/Reranker (`embedder_for`/`reranker_for`). Watcher gets a writer via `take_writer`/`release_writer` (reopens RO — Kuzu writer-visibility quirk). Idle-unloader thread evicts pooled models. Recovers Kuzu shadow pages (open RW once, replay, reopen RO). |
| `embed.py` | `sentence-transformers` wrapper. Process-wide `_MODEL_CACHE` keyed `(model, device, dtype)`. **Routes to the daemon** when client mode is on (chunked so `on_progress` ticks during indexing); transparent in-process fallback. CPU-fallback recovery on CUDA OOM/illegal-memory/cuBLAS/cuDNN. |
| `rerank.py` | Lazy `CrossEncoder`. Same `device=`/daemon-routing/CPU-fallback story as `Embedder`. |
| `daemon.py` | Optional shared embed+rerank daemon (loopback TCP). Owns one warm model each; serializes inference under one lock (the queue); two-stage idle (unload weights → exit to free context); lazy start. Client helpers route + lazily respawn (`ensure_daemon`). `configure_client` registers the spec at host startup. |
| `retrieve.py` | Hybrid retrieval + `explore`/`impact_of`/`test_impact`/`cypher`/`git_*`/`rules_for` + the analysis tools (`context`, `detect_changes`, `repo_map`, `list_clusters`/`cluster`, `route_map`/`api_impact`, `trace`, `health`, `symbol_history`, `rename_plan`, `processes`/`flow`, `index_info`). **All Cypher lives here or in `db.py`.** |
| `resolve.py` | Call-resolution confidence cascade (`ModuleIndex` import strings -> files, `SymbolResolver` tiers). Pure Python, fed by the indexer. |
| `frameworks.py` | Table-driven route / MCP-tool regexes (FastAPI, Flask, aiohttp, Starlette, Express, `@mcp.tool/resource/prompt`). Run by `parse.py`; hits on comment / docstring lines are dropped via the syntax tree. |
| `communities.py` | Louvain (networkx built-in) over calls + containment + imports; names, cohesion. |
| `insights.py` | Pure helpers: unified-diff parser, explainable risk score, token budgeting (`trim_sections`, `fit_to_budget`), Aider-style repo-map rendering, signatures. |
| `merkle.py` / `history.py` / `scip.py` | File-hash tree (stat reuse) / `git blame` symbol history + `git log -L` / dependency-free SCIP protobuf decoder + ingest. |
| `rename.py` / `agent_setup.py` / `bench.py` (+ `bench_questions.json`) | Apply a rename plan (API only, line-verified) / `agent-setup` skill + hook writer and the `hook` runtime / benchmark harness + runners. |
| `parse.py` | tree-sitter wrappers + tags queries. Method qname rescoping keyed on `id(node)`, not the qname string. Records call receivers (`extra.recv`/`attr`) and the module of each symbol import (`extra.module`/`alias`). |
| `watch.py` | `watchfiles` loop, N per-root tasks. Workspace `Semaphore(1)` serializes reindexes. Index uses the **pooled** embedder. |
| `mcp_tools.py` / `server.py` | 26 tools (15 retriever + `list_roots` + 11 analysis) + MCP resources (`docgraph://schema|clusters|cluster/{id}|flows|flow/{id}|routes`, root-scoped `docgraph://roots/{slug}/...`) + prompts (`detect_impact`, `architecture_map`); FastAPI + SSE + FastMCP at `/mcp`. **No `from __future__ import annotations`** (Pydantic can't resolve closure-local `RootSlug`). uvicorn needs `lifespan="on"` or `/mcp` 500s. |
| `rank.py` `git_tools.py` `rules.py` `llm.py` `wiki.py` `links.py` `fetch.py` `ignores.py` `summary.py` | PageRank / git-joined-to-graph / `.cursor/rules` + `AGENTS.md` matching / LLM client / module wiki / external-link crawl / 3-layer ignores / sub-function chunking. |

Runtime data: `<repo>/.docgraph/{graph.kuzu/, cache.json, state.json, merkle.json, repos.json, llm_docstrings.json, wiki/, scip-*.scip}`. `state.json` carries `schema_version`, `resolution` (edges per cascade tier), `embed_cache`, `scan`, `communities`, `scip`, `history_pending`, `removed_symbols`.

## Schema v3 (confidence, clusters, routes, history)

- **`db.SCHEMA_VERSION` gates the index format.** `index_all` compares it with `state.json["schema_version"]` and forces a full reindex on mismatch. Bump it whenever the DDL **or the cache.json entry shape** changes (e.g. new `RawEdge.extra` keys the resolver depends on). `GraphDB.migrate()` ALTERs added columns onto an old DB opened RW so queries never hard-error; readers probe `table_props()` / `Retriever._cap()` and degrade (`reindex_required`).
- **Resolution cascade** (`resolve.py`): import_map 0.95 > same_module 0.90 > import_suffix 0.85 > unique_global 0.75 > import_distance 0.55 > fuzzy 0.4/0.3 (common method names like `get` on a unique repo symbol: `common_name` 0.4). Receiver rules matter more than tiers: a call on an *external* module alias (`re.search`, `json.dumps`) resolves to nothing; a call on any other object can only hit methods; a bare call cannot hit a method in another file; fuzzy only for bare/self calls with non-CapWords names differing by underscores. Still-ambiguous sites write **CALLS_CANDIDATE** rows, never CALLS. SCIP edges are `scip` / 1.0 and upgrade heuristic edges in place.
- **Traversals are Python BFS** (`_bfs_calls` = one `frontier IN $ids` query per hop, confidence-filtered; `_mem_graph` = whole CALLS graph in memory for tools that fan out per symbol, e.g. `detect_changes` went 48 s -> 0.2 s). Don't reintroduce var-length Cypher for backward walks.
- **Communities, PageRank, Tier-4 are recomputed on every dirty run**, so Community ids change between index runs. Never persist them client-side.
- Kuzu reserves `max_db_size` of virtual address space per open Database (default 8 TB; Windows allows 128 TB per process). `GraphDB` passes `MAX_DB_SIZE = 1 TB`; without it ~16 simultaneously open DBs (many roots, or the test session) fail with `VirtualAlloc ... failed`.
- Kuzu: `RETURN n.*` is not supported -- `RETURN n AS n` gives a dict (with `_id`, `_label`). Unused query params: only pass params the query references.
- Routes/tools: `Route`/`Tool` nodes are per file (deleted with it); `HANDLES` to a handler in another file is re-linked through `db.framework_nodes()` when only the handler file changed.
- Embedding cache: vectors keyed by `ehash` = sha1(model + embed text). Incremental runs harvest vectors of the nodes they are about to delete, so moved/renamed/unchanged-body entities are not re-embedded. Full reindex does not use it.
- History: one `git blame --line-porcelain` per changed file. Files blamed with uncommitted lines go to `history_pending` and are re-blamed (SET in place) once HEAD moves -- a commit does not change a file hash, so the delta would never revisit them. `symbol_history` adds `git log -L` on demand.
- Embedding models: `embed.MODEL_PROFILES` holds dims, query/doc prefixes, `trust_remote_code` (only for models whose architecture is Hub code) and `max_seq`. `Embedder.embed(kind="query")` / `embed_query()` applies the query prefix; the daemon receives already-prefixed text. `jinaai/jina-embeddings-v2-base-code` cannot load on transformers >= 5 (its remote code); `nomic-ai/CodeRankEmbed` needs `pip install einops`.
- `agent-setup` writes only when invoked; `apply()` is idempotent (merges JSON hooks by the `docgraph hook pre-edit` marker, Markdown between `<!-- docgraph:begin/end -->`, git hooks between marker comments).

## Embedding daemon + GPU

GPU off by default; `--gpu` flips embedder to CUDA. `resolve_device(gpu)` returns `"cuda"` iff `torch.cuda.is_available()`. `Embedder.embed()` wraps inference in CPU-fallback recovery (don't remove it — see history). dtype default fp16 on CUDA, fp32 on CPU; override `--embed-dtype`. Embedding model = any HF sentence-transformers id; schema dim auto-derives from `dim_for_model()`. Switching dim on an existing DB is a hard error → `/api/admin/clear` + full reindex.

**Two model-lifecycle levels** (don't conflate — the old "restart the host on idle" reaper looped because it did):
1. **Idle-unload (weights)** — `--embed-idle-unload-sec` / `--rerank-idle-unload-sec`. Drops the model, `empty_cache()`, reloads lazily. **In-process, no restart.**
2. **Context-free (process exit)** — daemon-only `--idle-exit-sec`. After both models are unloaded and idle this long, the daemon **exits** to free the ~300 MB CUDA context; respawned on next demand. Loop-safe because the daemon does no GPU work on boot and is only respawned by an actual request.

**Daemon mode** (`docgraph host --embed-daemon`): one daemon owns embedder + reranker for the whole host; the host is GPU-stateless. Without it, models live in-process (pooled per host) with Level-1 unloading only; the context stays until the host exits. The indexer uses the **pooled** embedder (not a fresh one) so in-process sharing + daemon routing are uniform.

## Kuzu Cypher gotchas

- `label(r)` for rel type — **`type(r)` does not exist.** No `startNode`/`endNode`; use `(a)-[r]->(b)`.
- `File` nodes use `path`; every other entity uses `name`. `REFERENCES_` has a trailing underscore (`REFERENCES` is reserved).
- Bulk: `UNWIND $rows … CREATE` for nodes; `COPY <Edge> FROM arrow (from=,to=)` for edges (`_known_ids` filters dangling endpoints first — COPY hard-errors on missing PKs).
- A reader holds a lock that blocks writers — kill the host before `docgraph index`. Always `db.close()` before reopening RO; GC alone won't release the lock on Windows after COPY.
- **`UNWIND nodes(path)` on a backward var-length match with a bound end (`(caller)-[:CALLS*1..N]->(t {name: $x})`) segfaults Kuzu 0.11** on real graphs (N >= 2) and takes the whole host down — no exception to catch. Return the far endpoint instead (every intermediate node is itself an endpoint at a shorter depth). Forward matches from a bound start are fine. The tiny test fixture does not reproduce it.

## Per-file delta (the trickiest part — read `index.py::index_all`)

1. Hash-compare → changed/added/deleted/unchanged. 2. `DETACH DELETE` changed+deleted file nodes. 3. Re-parse only changed+added; update cache. 4. Symbol table from current DB state, not parse output. 5. Insert a cached edge only if `needs_insert(src,target)` (an endpoint just (re)created). 6. Tier-4 edges (`SIMILAR_TO`/`CO_CHANGED_WITH`/`TESTS`) + PageRank wiped+recomputed each run. Change parse-output shape → update cache writer **and** reader in lockstep. IDs: full reindex starts at 1; **incremental continues from `max(id)+1`** via `_seed_ids_from_db()`.

## Tree-sitter / conventions

- `tree-sitter >= 0.25`: `ts.Query(lang, src)` + `ts.QueryCursor(q).captures(node)` → `dict[str, list[Node]]`. One pip package per language; **don't** use `tree-sitter-language-pack 1.6+` (Rust rewrite, downloads grammars at runtime).
- Type-hint everything; `from __future__ import annotations` except `mcp_tools.py`/`server.py`. All Cypher in `db.py`/`retrieve.py`. Python 3.10 floor. No emojis / non-cp1252 chars in code, commits, or MCP docstrings (Windows console crashes).

## Things that have broken before — don't repeat

- DirectML embeddings `DXGI_ERROR_DEVICE_HUNG` / NVIDIA `nvwgf2umx.dll` segfault crashed the host → why we moved fastembed/ONNX → torch. Keep `Embedder.embed()`'s CPU-fallback wrapper.
- The "restart the host on idle to free the CUDA context" reaper **looped**: the indexer used a non-pooled embedder invisible to `models_status`, so it fired mid-index, hard-killed before the cache write, stranded a Kuzu WAL → every boot re-indexed the same files. Fixed by the daemon (idle-exit there, never the host) + pooled indexer + atomic cache writes. Don't reintroduce a host-restart-for-VRAM path.
- Non-cp1252 chars in MCP docstrings crash the call on Windows. `type(r)` in Cypher → error (use `label(r)`). `File.path` not `File.name`. Reading a Kuzu writer right after writing → empty (reopen RO). `del db; gc.collect()` won't release the Windows lock after COPY — call `db.close()`. Reasoning LLM endpoint without `reasoning_effort:"none"` → empty content. `str(enum_member)` on a `(str,Enum)` gives `'RootSlug.X'` — use `.value`. Use `is_user_ignored()` for files, `is_ignored()` only for dir pruning.

## Web UI (`docgraph/ui/index.html`)

One self-contained file, read fresh on every `GET /` (edit + reload, no restart). No CDN, no build. Hash routes `#/graph #/search #/wiki #/flows #/changes #/insights #/ask #/index` (phones: Wiki / Flows / Insights under the "More" tab); Ctrl K palette; tokens on `:root` with light/dark; breakpoints 1180 (detail pane → drawer) and 820 (bottom tab bar, explorer drawer, detail bottom sheet).
- **Every call goes through `request()` / `withRoot()`**, which appends `?root=<slug>` for the active root. Exception: `/api/jobs` — its `root` filter is a repo *path*, so the UI passes `{root:false}` and maps `job.root` back to a slug via `/api/roots`.
- **The graph page DOM is persistent** (built once; other pages render into `#pg-other`), so the canvas, worker layout and selection survive page switches. Engine: the Web Worker (`WORKER_SRC`, FA2-style + label propagation) is the same as the pre-redesign UI; render batches by colour, culls to the viewport and labels only the top-N by PageRank for the zoom level. Keep it that way — real roots have thousands of nodes.
- Graph load: `/api/graph?limit_nodes=50000` (all) or `/api/files` (level-of-detail "files first"); a click lazily merges `/api/node_neighbors` (1..3 hops). Colours come from CSS tokens via `readTheme()` — re-read on theme change.
- Colour-by-cluster uses the server `cluster` id when nodes carry it (worker label propagation is only the fallback for pre-v3 indexes); named hulls are recomputed at most every 400 ms for the top 40 clusters. The min-confidence slider filters CALLS edges on the canvas. The Changes page is driven by `detect_changes` and can push a diff overlay (changed = red ring, affected callers = amber) onto the graph. Flows draw inline-SVG sequence diagrams; Insights = health + clusters + trace + repo map. Rename shows the dry-run plan; "Apply graph edits" sits behind a typed-confirmation dialog.
- Controls the API cannot back are hidden or disabled with a reason (e.g. "Add root": roots are fixed for the host's lifetime). Never render placeholder data.
- Per-viewer state only in `localStorage` (theme, active root, pane widths, Ask threads, last Cypher query), always in try/catch.

## Testing

`.venv/Scripts/python -m pytest -p no:cacheprovider` (~310 tests). `test_index_html` only smoke-checks the UI; exercise UI changes in a real browser against a host on a spare port. Notable: `test_cli_flags` locks every flag telecode passes + the env-free contract; `test_daemon` exercises the daemon (ping/embed/rerank/status/idle-exit); `test_embed_fallback` the CUDA->CPU recovery; `test_workspace` the pool + shadow-page recovery; `test_graph_features` covers schema v3 on the `fw_indexed` fixture (`tests/fw_fixture.py`: routes for four frameworks, MCP tools, an ambiguous call, an external receiver, an import cycle, dead code, two commits, a synthetic SCIP index) plus incremental cache / merkle / history on a mutable copy. Kuzu writer-visibility: close the writer + reopen RO or test reads come back empty.

- Don't run pytest with `PYTHONIOENCODING=utf-8`: `test_cli_flags` decodes child `--help` output as cp1252 and rich's UTF-8 box characters then fail to decode.
- pytest's `addopts` already has `-q`; adding another `-q` hides the pass/fail summary line.
- A shell with `OMP_NUM_THREADS=1` makes CPU embedding ~10x slower (a 900-entity index took 7 min); use `--gpu` for scratch runs.
- Starting a host from `C:\Users\prith\.telecode` picks up telecode's own `docgraph/` package (`No module named docgraph.__main__`): set the working directory to this repo.

## Telecode integration

[Telecode](../.telecode) supervises one `docgraph host` for all roots and bridges its MCP tools as `docgraph_<tool>` (agents pass `root=<slug>` per call). It forwards every config value as a flag (incl. `--embed-daemon`/`--daemon-port`/`--daemon-idle-exit-sec`) and sweeps the daemon port on host stop (the daemon runs detached). Don't run `docgraph host`/stdio-mcp manually while telecode owns it. No docgraph-side code changes are needed for telecode — it just spawns the existing CLI.
