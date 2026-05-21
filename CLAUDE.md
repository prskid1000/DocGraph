# DocGraph — Working Notes for Claude

Things you can't infer from the code in 10 seconds. Read before non-trivial changes.

## What this is

Local code knowledge graph. Tree-sitter parses, sentence-transformers (torch) embeds, Kuzu stores, FastAPI + FastMCP serve. As of 2.2.0 a single `docgraph host` process serves **multiple roots** — one uvicorn, one MCP endpoint, one log, with a closed-enum `root` arg on every tool/route. v2 rewrite of an old Neo4j + ChromaDB + Streamlit + Vite stack (preserved at tag `v1-legacy` — don't resurrect).

## Hard rules

- **One package, one process, one DB file per root.** `docgraph host` is the unified server. No new top-level dirs, no separate frontend builds, no microservices.
- **Workspace is immutable for the host's lifetime.** Adding/removing roots requires a restart (telecode does this). No hot-reload endpoints.
- **Kuzu is the only data store.** No SQLite/Chroma/Neo4j/Redis.
- **No npm.** UI is one HTML file at `docgraph/ui/index.html`.
- **Embeddings via torch + sentence-transformers.** GPU is opt-in — `--gpu` flips the `Embedder` to `device="cuda"` and silently falls back to CPU if `torch.cuda.is_available()` returns False. Install the matching torch wheel from `https://download.pytorch.org/whl/cuXY` for CUDA support; the default PyPI wheel is CPU-only. Mac / AMD GPU users stay on CPU.
- **No env vars.** All config is `load_config(...)` kwargs / CLI flags. `docgraph` reads zero `DOCGRAPH_*` env vars.
- **Per-language processor classes are forbidden.** Add a language by adding two dict entries to `parse.py::LANGUAGES` + `TAGS_QUERIES`.
- **Tools/routes use a closed-enum `root` parameter** built from workspace slugs at host startup. Single root → enum has one defaulted value, LLM doesn't see it.

## File map

| File | Purpose |
|---|---|
| `cli.py` | Typer entry. `host` is the unified command; `serve`/`mcp`/`watch` delegate to it. Every config knob is a flag. |
| `config.py` | `load_config(repo_root, **overrides)` — fully kwarg-driven Config. Auto-detects ecosystems, respects `.gitignore` + `.docgraphignore`. `extra_roots` indexes multiple paths into ONE DB (different from workspace multi-root). `root_extra_paths(root)` / `save_root_extra_paths(root, paths)` r/w `<root>/.docgraph/repos.json`. `root_links(root)` / `save_root_links(root, links)` r/w `<root>/.docgraph/links.json`. |
| `links.py` | `ExternalLink` dataclass (`url`, `depth`, `max_pages`, `ttl_hours`, `last_fetched`, `page_count`). `load_links` / `upsert_link`. |
| `fetch.py` | BFS web crawler. `fetch_link(link, cancel_check, progress_cb)` — crawls up to `max_pages` pages at up to `depth` BFS levels. `fetch_all(cfg, cancel_check, progress_cb)` iterates all configured links. Progress callback signature: `(depth_level: int, done: int, total_at_depth: int)`. |
| `workspace.py` | `Workspace` registry of `RootSlot`s. Each slot owns its RO Kuzu connection + Retriever. Watcher gets a writer via `take_writer()` / `release_writer()` (which reopens RO — Kuzu writer-visibility quirk). 4-step `resolve()`: exact path → slug → file-prefix → default. |
| `parse.py` | tree-sitter wrappers + tags queries. Method qname rescoping is keyed on `id(node)` not on the qname string (sibling classes with same method name collided). |
| `index.py` | Parallel pipeline + per-file delta. **Most complex file.** |
| `db.py` | Kuzu schema + bulk insert. Edges via `COPY FROM arrow (from='X', to='Y')` (10-50× faster than MATCH+CREATE). `_known_ids` filters dangling endpoints before COPY (which hard-errors on missing PKs). `close()` is explicit; needed because Windows + Kuzu's COPY internals don't release the file lock on `del` alone. |
| `embed.py` | `sentence-transformers` wrapper over torch. Any HF sentence-transformers model works — schema dim auto-derives from `dim_for_model()` (static table + lazy probe for unknown models). `Embedder(device=...)` accepts `"cuda"` or `None`; re-checked against `torch.cuda.is_available()` at load time. Process-wide `_MODEL_CACHE` keyed on `(model_name, device, dtype)`. CPU-fallback recovery wraps `embed()` for CUDA OOM / illegal memory / driver crashes. |
| `rerank.py` | Lazy `sentence_transformers.CrossEncoder` (`jinaai/jina-reranker-v1-tiny-en`, ~33 MB) used when `search(rerank=True)`. Accepts `device=` for GPU; same CPU-fallback recovery as `Embedder`. |
| `retrieve.py` | Hybrid retrieval + `explore` / `impact_of` / `test_impact` / `cypher` / `git_*` / `rules_for`. **All Cypher lives here or in `db.py`.** |
| `rank.py` | NetworkX PageRank + `PersonalizedRanker` (cached graph). |
| `git_tools.py` | `git diff` / `blame` / `log`, joined to graph entities. |
| `rules.py` | `.cursor/rules/*.mdc` + `AGENTS.md` / `CLAUDE.md` glob-matching. |
| `llm.py` | urllib client for OpenAI/Anthropic-compatible local servers. Sends `reasoning_effort: "none"` so reasoning models skip thinking. Prompt overrides via `set_docstring_prompt(text)`. |
| `wiki.py` | LLM module wiki via `build_wiki(cfg, db, llm)`. Prompt-tail override via `set_wiki_prompt_tail(text)`. Falls back to a fact-sheet rendering when LLM unreachable. |
| `watch.py` | `watchfiles` loop. `watch_workspace` runs N async per-root tasks. Workspace-wide `Semaphore(1)` serializes reindexes. SSE `reindex_done {repo_slug, ts, events}`. |
| `mcp_tools.py` | 15 retriever tools + `list_roots`. Dynamic `RootSlug` enum from workspace slugs. **No `from __future__ import annotations`** — Pydantic can't resolve closure-local enums otherwise. |
| `mcp_stdio_proxy.py` | Strict stdio↔HTTP proxy for editors. Probes a running host first; refuses if scope path isn't a registered root. `--standalone` opts out. |
| `server.py` | FastAPI + SSE `/api/events` + FastMCP at `/mcp` (single port; uvicorn must use `lifespan="on"` or `/mcp` 500s). Same enum-evaluation concern → no future annotations. |
| `daemon.py` | Optional internal embedding daemon. **No CLI anymore** (host owns the embedder pool). `_serve_one()` calls `embedder._ensure().embed(...)` directly to avoid recursion through the daemon-aware wrapper. |
| `ui/index.html` | Canvas 2D viewer. Worker runs ForceAtlas2-lite + label-propagation communities. Multi-root brand pill swaps to a `<select>` when `/api/roots` returns >1; threads `?root=<slug>` into every fetch. |

Runtime data: `<repo>/.docgraph/{graph.kuzu/, cache.json, repos.json, llm_docstrings.json, wiki/}`.

## Kuzu Cypher gotchas

- Use `label(r)` for relationship type. **`type(r)` does not exist.**
- `startNode(r)` / `endNode(r)` don't exist either. Use `(a)-[r]->(b)` and reference `a`/`b`.
- `nodes(path)` works on variable-length paths; `relationships(path)` doesn't on all versions — avoid.
- `File` nodes use `path`. Every other entity uses `name`.
- `REFERENCES_` (trailing underscore) — `REFERENCES` is reserved.
- Edge tables can declare multiple FROM/TO pairs in one statement (see `db.py::EDGE_DDL`).
- Bulk insert: `UNWIND $rows AS row CREATE (n:L {...})` for nodes; `COPY <Edge> FROM arrow (from='X', to='Y')` for edges.
- A reader connection holds a lock that blocks writers. Kill the host before `docgraph index` or you'll see `"Could not set lock on file"`.
- Always call `db.close()` before reopening read-only. On Windows after COPY FROM, GC alone leaves the lock held.

## Tree-sitter API

- `tree-sitter >= 0.25`. Use `ts.Query(lang, src)` + `ts.QueryCursor(q).captures(node)` — not `lang.query(...)` from older docs.
- `captures()` returns `dict[str, list[Node]]`, not the old tuple list.
- One pip package per language. The TS package exposes both `language_typescript()` and `language_tsx()`.
- Don't switch to `tree-sitter-language-pack` 1.6+ — Rust-native rewrite that downloads grammars at runtime.

## ID allocation

Indexer uses int `id` PK per entity. Full reindex starts at 1. **Incremental must continue from `max(id) + 1`** via `_seed_ids_from_db()`. Don't break this.

## Per-file delta correctness

The trickiest part. Read `index.py::index_all` carefully. Contract:

1. Hash-compare → bucket files into changed/added/deleted/unchanged.
2. `DETACH DELETE` changed + deleted file nodes (Kuzu drops incident edges in the same step).
3. Re-parse only changed + added. Update cache.
4. Build symbol table from current DB state, not from parse output.
5. For each cached `RawEdge`: insert only if `needs_insert(src_file, target_file)` — at least one endpoint just (re)created.
6. Tier 4 edges (`SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`) and PageRank are wiped + recomputed every run.

If you change parse output shape, update cache writer **and** reader in lockstep.

## Testing

```bash
.venv/Scripts/python -m pytest                 # ~90s, ~250 tests
```

Files: `test_unit`, `test_indexer`, `test_retrieval`, `test_new_tools`, `test_multi_repo`, `test_cursor_parity`, `test_round3`, `test_api`, `test_mcp_server`, `test_llm` (mocked), `test_llm_live` (auto-skips if no LLM at `localhost:1235`), `test_daemon`, `test_workspace`, `test_cli_flags` (locks every flag telecode passes + the env-free contract), `test_embed_fallback` (torch CUDA OOM / cuBLAS / cuDNN → CPU recovery).

**Kuzu writer-visibility:** a writer connection doesn't see its own writes via subsequent `fetch_all`. Conftest closes the writer and reopens RO. Empty test results = forgot this.

**Full-reindex swap:** `Indexer.index_all(incremental=False)` swaps `self.db` with a fresh `GraphDB`. Close `indexer.db` (the active write-holder), not the original handle, before reopening RO.

## MCP tool surface — 15 retriever tools + `list_roots`

Base: `search`, `definition`, `references`, `call_graph`, `file_map`, `neighborhood`.

Differentiators: `explore`, `impact_of`, `test_impact`, `cypher` (read-only), `git_changes`, `git_blame`, `git_recent`, `rules_for`, `search(rerank=True)`, `list_roots`.

Every retriever-backed tool ends with `root: RootSlug` typed as a dynamic `(str, Enum)` built from workspace slugs at boot. Single-root → enum has one default value (LLM doesn't see it). Multi-root → LLM picks from a closed set; protocol rejects typos. Don't add tools without a strong reason — `search` accepts `focus_file` / `focus_symbol` for personalized PageRank.

## Multi-root

`Workspace.resolve(value)` is 4-step: None/"" → default; exact path; slug (case-insensitive); path-prefix. MCP/API callers send slugs (closed enum). Path-prefix is for direct Python callers + the stdio proxy.

`Config.extra_roots` is a different concept: indexes multiple paths into ONE DB (monorepo with sibling projects). Workspace roots are independent indexes side by side.

## Cancellation

Long ops (`/api/admin/index`, `/api/wiki/build`) are cooperatively cancellable via `POST /api/admin/cancel?root=<slug>`. One `CancelToken` per root in `cancel.py`. Long ops call `token.raise_if_set()` at phase boundaries (parse / embed / symbol-table / edge / tier-4 / pagerank) — never mid-Kuzu-COPY or mid-torch-forward, those would corrupt state. Routes translate `OperationCancelled` to HTTP 499. Telecode POSTs cancel **before** dropping the asyncio task, otherwise the connection drops first and the server never sees the signal. New long ops: same pattern (`reset_cancel` → `cancel_token_for` → pass in → `except OperationCancelled` → 499).

## Watch mode

`docgraph host --watch <root>` per-root watcher tasks on the same loop. Each takes a writer, runs `Indexer.index_all(incremental=True)` via `to_thread`, releases (which reopens RO). Workspace-wide `Semaphore(1)` serializes reindexes. Pre-debounce `_WatchFilter` per root drops `node_modules` / `.git` / non-source paths before `awatch` emits them.

## Ignore architecture

Three layers: (1) **Universal** baseline (VCS / OS junk / lockfiles / env / binaries, plus unambiguously-named dep dirs, **plus documentation-only files**: `README*`, `CHANGELOG*`, `LICENSE*`, `CONTRIBUTING*`, `AUTHORS*`, `CODEOWNERS`); (2) **Ecosystem autodetect** via marker files (`package.json`, `pom.xml`, `Cargo.toml`, `*.csproj`, ...) → unions in templates with ambiguously-named build dirs (`target/`, `build/`, `dist/`, ...); (3) **User files** (`.gitignore` / `.docgraphignore` / `.cursorindexingignore` exclude; `.cursorignore` indexes-but-redacts). Extra paths (`repos.json`) honour the same three layers — `_wire_extra_paths` mirrors `Config.__post_init__` to load user ignore files from within each extra path's directory.

`Config.is_ignored()` covers 1+2+user-exclude. `Config.is_ai_blocked()` / `ai_blocked_logical()` cover the AI-block tier.

Adding an ecosystem: add to `TEMPLATES` + `_DETECTORS` in `ignores.py`.

## Sub-function chunking

`summary.chunk_body(body, language=None)` splits entities > 1500 chars at scope boundaries (regex per language) once buffer crosses `CHUNK_TARGET_CHARS` (700). At hard cap (`CHUNK_MAX_CHARS` = 1400), mid-body split with overlap. Scope-aware flushes drop overlap so the next chunk starts cleanly. Stored as `Chunk` nodes with `CONTAINS_CHUNK` from parent. `Retriever._chunk_max_sims()` runs ALL chunk vectors against query once per search; entity score = `max(entity_sim, best_chunk_sim)`. Adding a language: drop a regex into `_SCOPE_BOUNDARY_PATTERNS`.

## LLM + GPU

LLM augmentation off by default; `--llm-model <name>` enables it. Talks to a local OpenAI-/Anthropic-compatible server (LM Studio / llama.cpp / vLLM / Ollama). Defaults: `localhost:1235`, openai. Every request carries `reasoning_effort: "none"` so reasoning models skip thinking — without it a 150-token budget comes back empty. Cache: `.docgraph/llm_docstrings.json` keyed by `sha256(body)`, rename-safe. Generated text is used **only** when no native docstring is present.

Embedding GPU off by default; `--gpu` enables it. Routes through torch via sentence-transformers: `resolve_device(cfg.gpu)` returns `"cuda"` if and only if `torch.cuda.is_available()`, else `None` (CPU). `Embedder._ensure()` re-checks at load time so a config asking for cuda on a CPU box silently downgrades. `Embedder.embed()` wraps inference in CPU-fallback recovery — CUDA OOM / illegal-memory / cuBLAS / cuDNN errors drop the cached model, force CPU, retry once.

Reranker GPU is independent: `cfg.rerank_gpu` / `--rerank-gpu`. Same `device=` + CPU-fallback story.

Embedding model: any HF sentence-transformers id via `--embed-model`. Schema dim auto-derives from `dim_for_model()` — static table for common models, lazy probe (load + read `get_sentence_embedding_dimension()`) for unknown ones. Switching dim on an existing DB is a hard error → `POST /api/admin/clear` + full reindex.

dtype: default is `fp16` on CUDA (~1.5–2× speedup, negligible cosine drift) and `fp32` on CPU. Override with `--embed-dtype bf16|fp32` if your GPU misbehaves on fp16.

Idle unload: per-class thresholds — `--embed-idle-unload-sec N` and `--rerank-idle-unload-sec N` (both default 0 = never). When either > 0 the workspace runs a periodic check (every 30s) and evicts pooled `Embedder` / `Reranker` torch sessions whose `last_used` is older than its respective threshold. Both are pooled on the workspace (`workspace.embedder_for(cfg)` / `workspace.reranker_for(cfg)`) so eviction is single-source. Reload is lazy on the next embed / score call. `unload()` calls `torch.cuda.empty_cache()` to release VRAM. Pairs well with telecode's `llamacpp.idle_unload_sec` for end-to-end model unloading.

## Coding conventions

- Type-hint everything. Use `from __future__ import annotations` except in `mcp_tools.py` / `server.py` (Pydantic + closure-local enums).
- All Cypher in `db.py` or `retrieve.py`.
- Python 3.10 floor.
- No emojis in code or commits unless asked. Windows + cp1252 console crashes on non-cp1252 chars in MCP tool docs.
- One short comment max above non-obvious blocks.

## Performance targets (don't regress)

| Scenario | Target |
|---|---|
| No-op incremental | < 0.05s |
| Touch (same hash) | < 0.05s |
| 1-file edit | < 2s |
| Full index, 100k LOC | < 10s on a modern laptop |
| MCP tool call | < 200ms typical |

## Common dev commands

```bash
.venv/Scripts/pip install -e .
.venv/Scripts/docgraph index --full
.venv/Scripts/docgraph host                                   # cwd as the only root → http://127.0.0.1:5500
.venv/Scripts/docgraph host --root /a --root /b --watch /a    # multi-root + watch
.venv/Scripts/docgraph stats
.venv/Scripts/docgraph mcp /a --transport stdio               # proxies the running host
taskkill //F //IM python.exe                                  # release DB lock (Windows)
```

### Bootstrap on a fresh Windows box

`setup.ps1` at the repo root creates `.venv`, installs torch from PyTorch's per-CUDA index (so the `+cuXY` wheel bundles its own CUDA + cuDNN runtime — no separate CUDA Toolkit install needed), runs `pip install -e .`, and drops a `docgraph.bat` shim into `~/.local/bin`. Flags: `-Recreate`, `-Python`, `-CudaVersion cu130|cu124|cpu` (default `cu130`), `-NoShim`, `-ShimDir`. The torch index URL is the only thing that changes between GPU/CPU installs — `pyproject.toml` declares plain `torch>=2.4`, and `setup.ps1` pre-seeds the right wheel before the editable install satisfies that dep against what's already installed.

The repo's `docgraph.bat` resolves the venv via `%~dp0` so the shim works from any clone location.

## Things that have broken before — don't repeat

- Non-cp1252 chars (`∈`) in MCP tool docstrings → crashes the call on Windows.
- `type(r)` in Cypher → `function TYPE does not exist`. Use `label(r)`.
- The daemon's embed-handler must NOT call `Embedder.embed()` (now daemon-aware → infinite recursion). Use `embedder._ensure().encode(...)` directly.
- Per-qname method rescoping must key on `id(def_node)`, not on the qname string — sibling classes collide otherwise.
- `File.path` is the name property, not `File.name`.
- Re-running the indexer with the host alive → DB lock error.
- `tree-sitter-language-pack 1.6+` is a Rust rewrite that downloads grammars at runtime. Stay on individual `tree-sitter-<lang>` packages.
- IDs starting at 1 on incremental → duplicate-PK error. Always `_seed_ids_from_db()` first.
- Reading from a Kuzu writer connection right after writing → empty results. Reopen RO.
- After `index_all(incremental=False)`, the original `GraphDB` is stale (`self.db` swapped). Close `indexer.db`.
- `del db; gc.collect()` doesn't release Kuzu's file lock on Windows after COPY FROM. Call `db.close()` explicitly.
- Reasoning-model endpoint without `reasoning_effort: "none"` → empty content.
- `from __future__ import annotations` in `mcp_tools.py` / `server.py` → Pydantic can't resolve the closure-local `RootSlug` enum.
- `str(enum_member)` on a `(str, Enum)` returns `'RootSlug.X'`, not `'x'`. Use `member.value`.
- Walking with `Config.is_ignored()` at the file level for the document pass → universal `*.pdf` / `*.png` swallow every asset before tier 3 sees it. Use `is_user_ignored()` for files; `is_ignored()` only for directory pruning.
- DirectML embeddings used to `DXGI_ERROR_DEVICE_HUNG` mid-inference under VRAM contention (and worse: the NVIDIA `nvwgf2umx.dll` segfault path that crashed the whole host process on driver bumps). Why this matters today: that's the reason we moved off fastembed/ONNX to torch + sentence-transformers. `Embedder.embed()` still has a CPU-fallback recovery wrapper — now around torch CUDA errors (OOM / illegal memory / cuBLAS / cuDNN). Don't remove it.
- Mounting FastMCP into FastAPI without `lifespan="on"` in uvicorn → `/mcp` 500s.
- Reading `os.environ` for config — there are no `DOCGRAPH_*` env vars anymore. Take a kwarg or a CLI flag.

## Telecode integration

[Telecode](../.telecode)'s tray supervises a single `docgraph host` child for all configured roots and bridges its MCP tools as `docgraph_<tool>` (no per-root prefix — agents pass `root=<slug>` per call).

- One child, one port, one log. `docgraph host --root A --root B [--watch …] --port <N>` + flags for every config value.
- Don't run `docgraph host` / stdio-mcp manually while telecode owns it — they'll contend for the port. Use `docgraph mcp <path> --transport stdio` (proxies the host) or `--standalone` for an isolated process.
- Auto-start: `docgraph.host.auto_start: true` in telecode's `settings.json`.

No code changes needed on the docgraph side — telecode just spawns the existing `host` CLI. Pointer: `<telecode>/docgraph/` package + the **DocGraph integration** section in `<telecode>/CLAUDE.md`.

## External links

`_maybe_fetch_links(cfg, force, cancel_check, progress_cb)` in `index.py` runs at the start of every index pass. It calls `fetch_all(cfg, cancel_check, progress_cb)` which iterates `<root>/.docgraph/links.json`. For each link older than `ttl_hours`, `fetch_link` runs a BFS crawl:

- `depth=0` → seed page only (single HTTP GET, no link extraction)
- `depth=1` → seed + all links found on the seed page
- `max_pages=N` → stop after N total pages saved (0 = unlimited)
- Cancel is propagated: `cancel_check` is called before each page fetch; `OperationCancelled` bubbles up and aborts the whole crawl
- Progress: `progress_cb(depth_level, done, total_at_depth)` is called after each page; tray renders `[1/12] fetching · level N · done/total`

Fetched pages are written to `<root>/.docgraph/fetched/<slug>/` and indexed as `File` nodes in the normal parse pass.

## Known limitations / next-up

- No SCIP / LSP integration → `CALLS` is name-based; mis-resolves overloads.
- Embedding model loads fresh per process (~1s cold). In-process duplication solved via `_MODEL_CACHE`.
- `IMPORTS_SYMBOL` is extracted from `@import.symbol` captures (Python `from x import Y`, JS/TS named imports, JS default-import shape). Java's qualified imports terminate in the symbol name; resolver matches the final dotted segment.
- `OVERRIDES` is derived in `index.py` from `INHERITS` + same-name methods via the inheritance closure (grandchild override of grandparent is recorded).
