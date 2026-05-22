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
| `retrieve.py` | Hybrid retrieval + `explore`/`impact_of`/`test_impact`/`cypher`/`git_*`/`rules_for`. **All Cypher lives here or in `db.py`.** |
| `parse.py` | tree-sitter wrappers + tags queries. Method qname rescoping keyed on `id(node)`, not the qname string. |
| `watch.py` | `watchfiles` loop, N per-root tasks. Workspace `Semaphore(1)` serializes reindexes. Index uses the **pooled** embedder. |
| `mcp_tools.py` / `server.py` | 15 retriever tools + `list_roots`; FastAPI + SSE + FastMCP at `/mcp`. **No `from __future__ import annotations`** (Pydantic can't resolve closure-local `RootSlug`). uvicorn needs `lifespan="on"` or `/mcp` 500s. |
| `rank.py` `git_tools.py` `rules.py` `llm.py` `wiki.py` `links.py` `fetch.py` `ignores.py` `summary.py` | PageRank / git-joined-to-graph / `.cursor/rules` + `AGENTS.md` matching / LLM client / module wiki / external-link crawl / 3-layer ignores / sub-function chunking. |

Runtime data: `<repo>/.docgraph/{graph.kuzu/, cache.json, state.json, repos.json, llm_docstrings.json, wiki/}`.

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

## Per-file delta (the trickiest part — read `index.py::index_all`)

1. Hash-compare → changed/added/deleted/unchanged. 2. `DETACH DELETE` changed+deleted file nodes. 3. Re-parse only changed+added; update cache. 4. Symbol table from current DB state, not parse output. 5. Insert a cached edge only if `needs_insert(src,target)` (an endpoint just (re)created). 6. Tier-4 edges (`SIMILAR_TO`/`CO_CHANGED_WITH`/`TESTS`) + PageRank wiped+recomputed each run. Change parse-output shape → update cache writer **and** reader in lockstep. IDs: full reindex starts at 1; **incremental continues from `max(id)+1`** via `_seed_ids_from_db()`.

## Tree-sitter / conventions

- `tree-sitter >= 0.25`: `ts.Query(lang, src)` + `ts.QueryCursor(q).captures(node)` → `dict[str, list[Node]]`. One pip package per language; **don't** use `tree-sitter-language-pack 1.6+` (Rust rewrite, downloads grammars at runtime).
- Type-hint everything; `from __future__ import annotations` except `mcp_tools.py`/`server.py`. All Cypher in `db.py`/`retrieve.py`. Python 3.10 floor. No emojis / non-cp1252 chars in code, commits, or MCP docstrings (Windows console crashes).

## Things that have broken before — don't repeat

- DirectML embeddings `DXGI_ERROR_DEVICE_HUNG` / NVIDIA `nvwgf2umx.dll` segfault crashed the host → why we moved fastembed/ONNX → torch. Keep `Embedder.embed()`'s CPU-fallback wrapper.
- The "restart the host on idle to free the CUDA context" reaper **looped**: the indexer used a non-pooled embedder invisible to `models_status`, so it fired mid-index, hard-killed before the cache write, stranded a Kuzu WAL → every boot re-indexed the same files. Fixed by the daemon (idle-exit there, never the host) + pooled indexer + atomic cache writes. Don't reintroduce a host-restart-for-VRAM path.
- Non-cp1252 chars in MCP docstrings crash the call on Windows. `type(r)` in Cypher → error (use `label(r)`). `File.path` not `File.name`. Reading a Kuzu writer right after writing → empty (reopen RO). `del db; gc.collect()` won't release the Windows lock after COPY — call `db.close()`. Reasoning LLM endpoint without `reasoning_effort:"none"` → empty content. `str(enum_member)` on a `(str,Enum)` gives `'RootSlug.X'` — use `.value`. Use `is_user_ignored()` for files, `is_ignored()` only for dir pruning.

## Testing

`.venv/Scripts/python -m pytest` (~90s, ~250 tests). Notable: `test_cli_flags` locks every flag telecode passes + the env-free contract; `test_daemon` exercises the daemon (ping/embed/rerank/status/idle-exit); `test_embed_fallback` the CUDA→CPU recovery; `test_workspace` the pool + shadow-page recovery. Kuzu writer-visibility: close the writer + reopen RO or test reads come back empty.

## Telecode integration

[Telecode](../.telecode) supervises one `docgraph host` for all roots and bridges its MCP tools as `docgraph_<tool>` (agents pass `root=<slug>` per call). It forwards every config value as a flag (incl. `--embed-daemon`/`--daemon-port`/`--daemon-idle-exit-sec`) and sweeps the daemon port on host stop (the daemon runs detached). Don't run `docgraph host`/stdio-mcp manually while telecode owns it. No docgraph-side code changes are needed for telecode — it just spawns the existing CLI.
