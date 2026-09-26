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
- **Per-language processor classes are forbidden.** Add a language = dict entries in `parse.py`: `EXT_TO_LANG` (or `FILENAME_TO_LANG` / `FILENAME_PREFIX_TO_LANG` for `Dockerfile`-style names) + `LANGUAGES` + `TAGS_QUERIES`. A missing / broken wheel falls back to the plain-text path automatically; `TEXT_EXT_KINDS` / `TEXT_NAME_KINDS` only name the text kind.
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
| `communities.py` | Louvain (networkx built-in) over calls + containment + imports; `detect_arrays` aggregates in numpy and folds symbols -> files above `MAX_FINE_NODES`, files -> directories above `MAX_FOLD_NODES`; names, cohesion. |
| `insights.py` | Pure helpers: unified-diff parser, explainable risk score, token budgeting (`trim_sections`, `fit_to_budget`), Aider-style repo-map rendering, signatures. |
| `merkle.py` / `history.py` / `scip.py` | File-hash tree (stat reuse) / `git blame` symbol history + `git log -L` / dependency-free SCIP protobuf decoder + ingest. |
| `rename.py` / `agent_setup.py` / `bench.py` (+ `bench_questions.json`) | Apply a rename plan (API only, line-verified) / `agent-setup` skill + hook writer and the `hook` runtime / benchmark harness + runners. |
| `parse.py` | tree-sitter wrappers + tags queries (40+ grammars). `classify_file` decides grammar / `text:<kind>` / skip (binary ext, NUL or undecodable sniff of the first 8 KB). Plain-text files and grammar files without any function/class get file-level `chunks` (`summary.text_chunks`, line/paragraph-aware); `.ipynb` code cells parse as the kernel language over a virtual text (`notebook_text`, also what `/api/file_content` returns), markdown cells become chunks. Method qname rescoping keyed on `id(node)`, not the qname string. Records call receivers (`extra.recv`/`attr`) and the module of each symbol import (`extra.module`/`alias`). |
| `similar.py` | Bounded top-k cosine for `SIMILAR_TO`: exact row blocks up to 8k rows, IVF lists (k-means, oversized lists split) above. Never an n x n matrix. |
| `layout.py` | Deterministic hierarchical world layout (symbols spiral in their file disc, files in their cluster, clusters in the world); `place_file` / `spiral_offsets` for incremental placement. Pure numpy. |
| `tiles.py` | LOD tile sidecar (`.docgraph/tiles/`): `build()` from arrays, `TileStore` serves packed binary tiles (`tile`, `locate`, `frame_batch`). The format is mirrored by `parseTile` in the UI and `_decode` in `tests/test_scale.py` -- change all three together. |
| `watch.py` | `watchfiles` loop, N per-root tasks. Workspace `Semaphore(1)` serializes reindexes. Index uses the **pooled** embedder. |
| `mcp_tools.py` / `server.py` | 26 tools (15 retriever + `list_roots` + 11 analysis) + MCP resources (`docgraph://schema|clusters|cluster/{id}|flows|flow/{id}|routes`, root-scoped `docgraph://roots/{slug}/...`) + prompts (`detect_impact`, `architecture_map`); FastAPI + SSE + FastMCP at `/mcp`. **No `from __future__ import annotations`** (Pydantic can't resolve closure-local `RootSlug`). uvicorn needs `lifespan="on"` or `/mcp` 500s. |
| `rank.py` (scipy sparse power iteration, `ScoreMap`) `git_tools.py` `rules.py` `llm.py` `wiki.py` `links.py` `fetch.py` `ignores.py` `summary.py` | PageRank / git-joined-to-graph / `.cursor/rules` + `AGENTS.md` matching / LLM client / module wiki / external-link crawl / 3-layer ignores / sub-function chunking. |

Runtime data: `<repo>/.docgraph/{graph.kuzu/, cache.json, state.json, merkle.json, repos.json, llm_docstrings.json, wiki/, scip-*.scip, tiles/}`. `state.json` carries `schema_version`, `resolution` (edges per cascade tier), `resolution_scope`, `embed_cache`, `scan`, `communities`, `scip`, `search_index`, `analytics` (+ `analytics_drift`), `tiles` / `tiles_generation`, `history_pending`, `removed_symbols`.

## Schema v3 (confidence, clusters, routes, history)

- **`db.SCHEMA_VERSION` gates the index format.** `index_all` compares it with `state.json["schema_version"]` and forces a full reindex on mismatch. Bump it whenever the DDL **or the cache.json entry shape** changes (e.g. new `RawEdge.extra` keys the resolver depends on). `GraphDB.migrate()` ALTERs added columns onto an old DB opened RW so queries never hard-error; readers probe `table_props()` / `Retriever._cap()` and degrade (`reindex_required`).
- **Resolution cascade** (`resolve.py`): import_map 0.95 > same_module 0.90 > import_suffix 0.85 > unique_global 0.75 > import_distance 0.55 > fuzzy 0.4/0.3 (common method names like `get` on a unique repo symbol: `common_name` 0.4). Receiver rules matter more than tiers: a call on an *external* module alias (`re.search`, `json.dumps`) resolves to nothing; a call on any other object can only hit methods; a bare call cannot hit a method in another file; fuzzy only for bare/self calls with non-CapWords names differing by underscores. Still-ambiguous sites write **CALLS_CANDIDATE** rows, never CALLS. SCIP edges are `scip` / 1.0 and upgrade heuristic edges in place.
- **Traversals are Python BFS** (`_bfs_calls` = one `frontier IN $ids` query per hop, confidence-filtered; `_mem_graph` = whole CALLS graph in memory for tools that fan out per symbol, e.g. `detect_changes` went 48 s -> 0.2 s). Don't reintroduce var-length Cypher for backward walks.
- **Communities / PageRank / layout** (see Schema v4 below): graphs up to `full_recompute_max_nodes` recompute PageRank + communities on every dirty run, so Community ids change between runs -- never persist them client-side. Tier-4 `SIMILAR_TO` / `TESTS` are partial on incrementals.
- Kuzu reserves `max_db_size` of virtual address space per open Database (default 8 TB; Windows allows 128 TB per process). `GraphDB` passes `MAX_DB_SIZE = 1 TB`; without it ~16 simultaneously open DBs (many roots, or the test session) fail with `VirtualAlloc ... failed`.
- Kuzu: `RETURN n.*` is not supported -- `RETURN n AS n` gives a dict (with `_id`, `_label`). Unused query params: only pass params the query references.
- Routes/tools: `Route`/`Tool` nodes are per file (deleted with it); `HANDLES` to a handler in another file is re-linked through `db.framework_nodes()` when only the handler file changed.
- Embedding cache: vectors keyed by `ehash` = sha1(model + embed text). Incremental runs harvest vectors of the nodes they are about to delete, so moved/renamed/unchanged-body entities are not re-embedded. Full reindex does not use it.
- History: one `git blame --line-porcelain` per changed file. Files blamed with uncommitted lines go to `history_pending` and are re-blamed (SET in place) once HEAD moves -- a commit does not change a file hash, so the delta would never revisit them. `symbol_history` adds `git log -L` on demand.
- Embedding models: `embed.MODEL_PROFILES` holds dims, query/doc prefixes, `trust_remote_code` (only for models whose architecture is Hub code) and `max_seq`. `Embedder.embed(kind="query")` / `embed_query()` applies the query prefix; the daemon receives already-prefixed text. `jinaai/jina-embeddings-v2-base-code` cannot load on transformers >= 5 (its remote code); `nomic-ai/CodeRankEmbed` needs `pip install einops`.
- `agent-setup` writes only when invoked; `apply()` is idempotent (merges JSON hooks by the `docgraph hook pre-edit` marker, Markdown between `<!-- docgraph:begin/end -->`, git hooks between marker comments).

## Schema v4 (text fallback, search indexes, world layout, tiles)

- **`SCHEMA_VERSION = 4`**: FLOAT embeddings, `Chunk.{line_start,line_end,terms}`, `CONTAINS_CHUNK` from File, `terms` + `x`/`y` on symbols, `Community.{x,y,r}`. A v3 DB forces one full reindex.
- **Search indexes**: Kuzu 0.11 statically links VECTOR and FTS (`show_loaded_extensions`), so nothing is INSTALLed and it works offline and read-only. `GraphDB.ensure_search_indexes()` creates `fn_vec cls_vec chunk_vec` (HNSW, cosine) + `fn_fts cls_fts chunk_fts` after the bulk load of a full pass; Kuzu maintains both on every CREATE / DELETE, so incrementals do nothing. `Retriever.search()` takes per-label top-k from both + an exact-name probe + chunk parents and scores only those; `search_backend()` == "brute" (pre-v4 DB) runs the old whole-table path (`_search_brute`).
- **Streamed indexing**: steps 2-5 run per batch of `index_batch_files`; the cache keeps slim entities (no bodies), read / written with orjson.
- **Analytics** (`Indexer._graph_analytics`): one `_global_pass` reads nodes (`db.layout_source`) and edges (`db.edge_endpoints`) once as Arrow / numpy and derives PageRank, communities, layout and tiles. Above `full_recompute_max_nodes` an incremental pass patches rank / communities for the changed files (`_patch_pagerank`, `_assign_new_communities`) until `analytics_drift` exceeds `recompute_drift` x files. Layout is incremental for every size (`_place_new`: a changed file keeps its harvested centre, its symbols re-spiral); a global relayout happens only on a full pass, on drift, or when positions are missing. Positions live in Kuzu (`x`, `y`); the tile sidecar is derived and rebuilt at the end of every dirty pass.
- **Incremental resolution scope**: an unchanged file's edge is only resolved when its target name (direct, alias original, or `_norm` fuzzy key) names an entity of a changed file -- exactly the edges `needs_insert` could accept. A new resolution tier that matches by something other than the name must extend that filter (`hot_names` in `index_all`).
- **Tiles**: every LOD (`sym`, `file`, `clu`) is one Morton-sorted array set, so any quadtree tile is a code range (no per-level copies). Per tile the server serves the finest LOD whose node count fits `budget`; the finest level always serves `sym` -- nothing is ever truncated. Edges are stored once under the lower-id endpoint plus a reverse index (a tile also serves edges it only receives); foreign endpoints travel as ghost endpoints. The bbox is padded 12.5% so incremental placements stay inside it.
- Tile arrays are loaded into RAM (`np.load` without mmap): a memory-mapped `.npz` on Windows would block the next build's `os.replace`.

## Embedding daemon + GPU

GPU off by default; `--gpu` flips embedder to CUDA. `resolve_device(gpu)` returns `"cuda"` iff `torch.cuda.is_available()`. `Embedder.embed()` wraps inference in CPU-fallback recovery (don't remove it — see history). dtype default fp16 on CUDA, fp32 on CPU; override `--embed-dtype`. Embedding model = any HF sentence-transformers id; schema dim auto-derives from `dim_for_model()`. Switching dim on an existing DB is a hard error → `/api/admin/clear` + full reindex.

**Two model-lifecycle levels** (don't conflate — the old "restart the host on idle" reaper looped because it did):
1. **Idle-unload (weights)** — `--embed-idle-unload-sec` / `--rerank-idle-unload-sec`. Drops the model, `empty_cache()`, reloads lazily. **In-process, no restart.**
2. **Context-free (process exit)** — daemon-only `--idle-exit-sec`. After both models are unloaded and idle this long, the daemon **exits** to free the ~300 MB CUDA context; respawned on next demand. Loop-safe because the daemon does no GPU work on boot and is only respawned by an actual request.

**Daemon mode** (`docgraph host --embed-daemon`): one daemon owns embedder + reranker for the whole host; the host is GPU-stateless. Without it, models live in-process (pooled per host) with Level-1 unloading only; the context stays until the host exits. The indexer uses the **pooled** embedder (not a fresh one) so in-process sharing + daemon routing are uniform.

## Kuzu Cypher gotchas

- `label(r)` for rel type — **`type(r)` does not exist.** No `startNode`/`endNode`; use `(a)-[r]->(b)`.
- `CALL table_info('<rel>')` returns **no rows for a rel table without properties** (CONTAINS, IMPORTS, MEMBER_OF ...): `GraphDB.has_table` falls back to `show_tables()`. Don't test existence with `table_props`.
- With a vector index on a column, `SET n.embedding = ...` raises ("used in one or more indexes") -- delete + insert instead (the indexer only ever does that).
- FTS does not split camelCase / snake_case identifiers -- that is what the `terms` column is for (`index._terms`). `QUERY_FTS_INDEX` rows are not sorted: always `ORDER BY score DESC`. `QUERY_VECTOR_INDEX` cannot take an UNWIND variable as its query vector (one call per vector).
- Bulk reads for analytics go through `GraphDB.fetch_arrow` / `_np` / `edge_endpoints` / `node_columns` (Arrow -> numpy); `fetch_all` builds a Python dict per row and is 10-50x slower at 1M rows.
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
- **The graph page DOM is persistent** (built once; other pages render into `#pg-other`), so the canvases, loaded tiles and selection survive page switches.
- **Graph engine = streamed tiles + WebGL** (no global node cap, no `/api/graph` download): `/api/tiles/manifest`, then quadtree tiles like map chunks. `levelFor(scale)` picks the level so a tile is about `tilePx()` on screen; `updateWanted` computes the visible set + a one-tile prefetch ring, queues nearest-first and aborts batches whose tiles all left the ring; `pump` runs <= 3 batch requests of <= 6 tiles; `evict` drops LRU tiles over `residentBudget()` (smaller on phones). Tiles crossfade by `alpha`. Cache API (`docgraph-tiles-v2`, keyed per root, tile and detail budget) + ETag revalidation when the manifest generation changes.
- Rendering: `glInit` (WebGL2, else WebGL1 + `ANGLE_instanced_arrays`, else Canvas 2D `draw2D`), one node + edge buffer per tile, rebuilt only when `G.styleVersion` changes (`restyle()` after any filter / colour / focus / theme change). `antialias:false` on purpose (circles are anti-aliased in the fragment shader; MSAA made software GL several times slower). Overlays are 2D canvases: `#cv-under` hulls (from File / cluster positions), `#cv-over` selection neighbourhood (from `/api/node_neighbors`, ghost rings for nodes outside loaded tiles), rings and labels (top PageRank in view, collision grid).
- Every node lookup goes through `lookup(id)` (resident tile -> neighbourhood -> positions from `/api/tiles/locate`). Search / focus / diff overlay jump with `centerOn`. Cluster super-nodes fly-to on click. The play button runs the worker refine on the loaded symbol tiles only (borders pinned); nothing it does is persisted.
- Colours come from CSS tokens via `readTheme()` — re-read on theme change. The Changes page can push a diff overlay (changed = red ring, affected callers = amber). Flows draw inline-SVG sequence diagrams; Insights = health + clusters + trace + repo map. Rename shows the dry-run plan; "Apply graph edits" sits behind a typed-confirmation dialog.
- `window.__dgStats` (fps, bytes, per-tile latency, first frame) and `G.forceLevel` exist for benchmarks.
- Controls the API cannot back are hidden or disabled with a reason (e.g. "Add root": roots are fixed for the host's lifetime). Never render placeholder data.
- Per-viewer state only in `localStorage` (theme, active root, pane widths, Ask threads, last Cypher query), always in try/catch.

## Testing

`.venv/Scripts/python -m pytest -p no:cacheprovider` (~420 tests; `test_languages.py` = one fixture per grammar, `test_scale.py` = text fallback / notebooks / indexed search + brute fallback / bounded similarity, PageRank, communities / layout / tiles + endpoints / incremental layout and resolution scope). `test_index_html` only smoke-checks the UI; exercise UI changes in a real browser against a host on a spare port. Notable: `test_cli_flags` locks every flag telecode passes + the env-free contract; `test_daemon` exercises the daemon (ping/embed/rerank/status/idle-exit); `test_embed_fallback` the CUDA->CPU recovery; `test_workspace` the pool + shadow-page recovery; `test_graph_features` covers schema v3 on the `fw_indexed` fixture (`tests/fw_fixture.py`: routes for four frameworks, MCP tools, an ambiguous call, an external receiver, an import cycle, dead code, two commits, a synthetic SCIP index) plus incremental cache / merkle / history on a mutable copy. Kuzu writer-visibility: close the writer + reopen RO or test reads come back empty.

- Don't run pytest with `PYTHONIOENCODING=utf-8`: `test_cli_flags` decodes child `--help` output as cp1252 and rich's UTF-8 box characters then fail to decode.
- pytest's `addopts` already has `-q`; adding another `-q` hides the pass/fail summary line.
- A shell with `OMP_NUM_THREADS=1` makes CPU embedding ~10x slower (a 900-entity index took 7 min); use `--gpu` for scratch runs.
- Starting a host from `C:\Users\prith\.telecode` picks up telecode's own `docgraph/` package (`No module named docgraph.__main__`): set the working directory to this repo.
- A script that runs `Indexer.index_all` must guard its body with `if __name__ == "__main__":` -- the parse pool spawns (Windows) re-import `__main__`, and an unguarded script re-opens the DB in every worker (`Could not set lock on file`).
- UI checks in headless Chrome: `--use-angle=swiftshader` is software GL (compositing-bound); add `--use-angle=d3d11 --enable-gpu` for the real GPU. A tab opened over CDP must be brought to front (`Page.bringToFront`) or rAF is throttled.

## Telecode integration

[Telecode](../.telecode) supervises one `docgraph host` for all roots and bridges its MCP tools as `docgraph_<tool>` (agents pass `root=<slug>` per call). It forwards every config value as a flag (incl. `--embed-daemon`/`--daemon-port`/`--daemon-idle-exit-sec`) and sweeps the daemon port on host stop (the daemon runs detached). Don't run `docgraph host`/stdio-mcp manually while telecode owns it. No docgraph-side code changes are needed for telecode — it just spawns the existing CLI.
