# DocGraph — Working Notes for Claude

This file captures the things you can't infer from the code in 10 seconds. Read it before making non-trivial changes.

## What this project is

A local code knowledge graph. Indexes any repo with tree-sitter, embeds entities with fastembed, and exposes everything via MCP tools and a single-page web UI. As of 2.2.0, a single host process serves **multiple roots** (repos) at once — one uvicorn, one MCP endpoint, one log, with a closed-enum `root` argument on every tool/route so agents pick which repo per call.

It is the **v2 rewrite** of an older Neo4j + ChromaDB + Streamlit + Vite stack. The old code is preserved at tag `v1-legacy`. Do not resurrect any of those dependencies.

## Hard rules

- **One Python package, one process per machine, one DB file *per root*.** The `docgraph host` command runs the unified server (web UI + JSON API + MCP HTTP + optional watchers) for any number of roots. No new top-level dirs, no separate frontend builds, no microservices.
- **Workspace is immutable for a host's lifetime.** Adding/removing roots requires a host restart (telecode does this automatically). This kept the design simple: no hot-reload endpoints, no synchronization on root membership, no dynamic schema regeneration mid-flight.
- **Kuzu is the only data store.** Don't add SQLite, ChromaDB, Neo4j, Redis, etc.
- **No npm.** The web UI is one HTML file at `docgraph/ui/index.html`. No build step.
- **No torch dependency.** Embeddings go through `fastembed` (ONNX). Adding sentence-transformers/torch would 4× the install size. GPU support uses ONNX Runtime providers (`CUDAExecutionProvider` / `DmlExecutionProvider` / `CoreMLExecutionProvider`) — not torch CUDA. Users opt in by `pip install onnxruntime-gpu` (or `-directml` / `-silicon`); we never depend on those.
- **Per-language processor classes are forbidden.** All language support comes from `parse.py::LANGUAGES` + `TAGS_QUERIES`. Adding a language = pip install `tree-sitter-<x>` and add two dict entries.
- **Tools/routes use a closed-enum `root` parameter.** Built dynamically from the workspace's slugs at host startup. The LLM picks from a known set; typos rejected at the protocol layer; no free-form path matching needed at request time. Single-root case: enum has one value and is the default — callers don't need to think about it.

## File map

| File | Purpose |
|---|---|
| `docgraph/cli.py` | Typer entry. `host` is the unified command (web UI + JSON API + MCP HTTP + optional watchers). `serve`/`mcp`/`watch` are thin shims that delegate to `host`. Each command takes positional `<path>` (single-root sugar) and/or repeatable `--root` flags (multi-root). |
| `docgraph/workspace.py` | `Workspace` registry holding a `RootSlot` per registered root. Each slot owns its read-only Kuzu connection + Retriever; the watcher gets a writer on demand via `take_writer()` and releases it via `release_writer()` after each reindex (the workspace then reopens the read-only handle so subsequent queries see new data — Kuzu's writer-visibility quirk). 4-step lookup in `resolve()`: exact path → slug → file-prefix → default. Embedder pool deduped by `(model_name, gpu)`. |
| `docgraph/config.py` | Auto-detects repo root + ecosystems, respects `.gitignore` + `.docgraphignore`. Single-root indexer-side multi-root via `Config.extra_roots` (different concept: indexes multiple paths into one DB). Persisted in `.docgraph/repos.json`. The host-level multi-root abstraction lives in `workspace.py`. |
| `docgraph/ignores.py` | Universal ignore patterns + per-ecosystem autodetect (Node / Python / Maven / Gradle / Rust / .NET / Swift / Ruby / Dart / Elixir / Scala / PHP / Terraform / Unity / Go). Universal layer also covers Jupyter / MLflow / wandb / DVC / Haskell / Zig / R / Scala tooling. Inline string lists, no template files. |
| `docgraph/parse.py` | tree-sitter wrappers + tags queries per language. Method qname rescoping is keyed on `id(node)` not on the original qname string — methods sharing a base qname (`file::area` for both `Square.area` and `Circle.area`) used to collide and remap to one final qname. |
| `docgraph/daemon.py` | Optional cross-CLI embedding daemon, kept as an internal optimization only — there is **no** `docgraph daemon` CLI command anymore (the host owns the embedder pool, so the daemon was redundant). The TCP shape (length-prefixed JSON on loopback, default 5577) is unchanged. `Embedder.embed()` consults `embed_via_daemon()` before loading its own ONNX session; daemon-side `_serve_one()` calls `embedder._ensure().embed(...)` directly to avoid recursing back through the wrapper. |
| `docgraph/index.py` | Parallel pipeline + per-file delta updates. **Most complex file.** |
| `docgraph/summary.py` | Builds the embedding text per entity (extracts docstrings/JSDoc/Rust `///` etc.). |
| `docgraph/db.py` | Kuzu schema + bulk insert helpers. Edges go through `COPY FROM arrow (from='X', to='Y')` (pyarrow-staged) — 10-50× faster than the old MATCH+CREATE. Default `insert_edges` batch is 10k rows. The `_known_ids` cache filters dangling-endpoint rows before COPY (which hard-errors on missing PKs, vs. the old silent-drop). `close()` is explicit + `__del__` calls it — needed because Windows + Kuzu's COPY internals don't release the file lock on `del` alone. |
| `docgraph/embed.py` | fastembed wrapper (BGE-small, 384-dim). `Embedder(providers=...)` passes ORT providers through; `GPU_PROVIDERS` is the default GPU stack (CUDA → DirectML → CoreML → ROCm → CPU). Process-wide `_MODEL_CACHE` (locked) keys on `(model_name, providers)` so multiple `Embedder()` instances share one loaded ONNX session — important for the test suite, multi-repo, and `watch --serve` reload paths. `clear_model_cache()` exposed for tests; production code never needs it. |
| `docgraph/rank.py` | NetworkX PageRank + `PersonalizedRanker` (query-time personalized PR with cached graph). |
| `docgraph/retrieve.py` | Hybrid retrieval + `explore` / `impact_of` / `test_impact` / `cypher` / `git_*` / `rules_for`. **All Cypher lives here or in `db.py`.** |
| `docgraph/git_tools.py` | `git diff` / `blame` / `log` shell-outs, joined to graph entities. |
| `docgraph/rules.py` | Parses `.cursor/rules/*.mdc` + `AGENTS.md` / `CLAUDE.md`; glob-matches per file. |
| `docgraph/rerank.py` | Lazy cross-encoder (`jinaai/jina-reranker-v1-tiny-en`, ~33 MB); used when `search(rerank=True)`. |
| `docgraph/llm.py` | Tiny urllib-based client for OpenAI- or Anthropic-compatible local servers. Used by `--llm-model <name>` to summarize entities lacking native docs. Off by default. Sends `reasoning_effort: "none"` so reasoning models (Qwen3, DeepSeek-R1) skip thinking and a 150-token budget fits a one-sentence answer. |
| `docgraph/docs.py` | URL fetch + HTML→text + chunking + Doc node ingestion. Cursor `@Docs` parity. |
| `docgraph/wiki.py` | LLM-grounded module wiki. `build_wiki(cfg, db, llm)` walks the top-level dirs, gathers a fact sheet from Kuzu (top classes / functions / importers / tests by PageRank), prompts the LLM, writes Markdown to `<repo>/.docgraph/wiki/<slug>.md`. Falls back to a fact-sheet rendering when the LLM is unreachable so the wiki is never blank. CLI: `docgraph wiki [--module X]`. API: `/api/wiki/list`, `/api/wiki/page?slug=`, `POST /api/wiki/build`. The fact-gathering uses `c.body` (not `c.docstring` — that property doesn't exist on the schema) and joins File via `path` not `file`. |
| `docgraph/watch.py` | `watchfiles` loop with pre-debounce ignore filter. `watch_workspace(workspace, roots)` runs N async `_watch_one` tasks (one per root) on the current event loop. `watch_and_serve_workspace` does the same but also boots uvicorn for the host app. A workspace-wide `asyncio.Semaphore(1)` serializes reindexes across roots so two CPU-bound passes don't fight. Each per-root task takes a writer, runs `Indexer.index_all(incremental=True)` via `to_thread()`, releases the writer (which reopens the workspace's RO slot), and broadcasts SSE `reindex_done {repo_slug, ts, events}`. |
| `docgraph/mcp_tools.py` | 15 retriever-backed tools + `list_roots`. Each retriever-backed tool gets a `root: RootSlug = DEFAULT` argument typed as a dynamic `(str, Enum)` built at `make_mcp` time from the workspace's slugs. **No `from __future__ import annotations` here** — Pydantic's lazy `get_type_hints` can't resolve the closure-local `RootSlug` if annotations are stringified. |
| `docgraph/mcp_stdio_proxy.py` | Strict stdio↔HTTP proxy for editors (Cursor, Claude Desktop). Probes `http://127.0.0.1:5500/api/roots`; if a host is up, exposes a stdio surface scoped to the editor's `<path>` (refuses if that path isn't a registered root — config-consistency over silent duplication). If no host is reachable, exits with a clear "start a host first" error. `--standalone` is the explicit opt-out for "I don't want host sharing". |
| `docgraph/server.py` | FastAPI host: web UI + JSON API + SSE `/api/events` + FastMCP mounted at `/mcp` (single port, lifespan chained — uvicorn config must use `lifespan="on"` or the `/mcp` route 500s). `make_app(workspace)` builds a dynamic `RootSlug` enum from the workspace's slugs and threads it through every route as a closed-enum `root` query param. Discovery / admin: `GET /api/roots` (lists registered roots), `POST /api/admin/index` (in-process incremental or `{full:true}` reindex via `Workspace.take_writer / release_writer`; lets external supervisors avoid Kuzu's exclusive writer lock). **No `from __future__ import annotations`** — same enum-evaluation concern as `mcp_tools.py`. |
| `docgraph/ui/index.html` | Single-page graph viewer. Canvas 2D only — Sigma.js / WebGL was removed in 2.1.0 because it was unreliable on first paint. Performance comes from a Web Worker that runs **ForceAtlas2-lite** (linear repulsion, degree-weighted, on a spatial grid) + **label-propagation community detection**. Worker bootstrap is an inline `Blob([WORKER_SRC])`-backed Worker — no esm.sh, no CDN. Render batches edges/nodes by color and viewport-culls in world coords. New tabs: Processes (entry-point → call chains via `/api/processes`) and Wiki (LLM module pages via `/api/wiki/*`). Color-mode toggle (kind / community) lives in the Filters tab. |

Runtime data: `<repo>/.docgraph/graph.kuzu/` (DB), `<repo>/.docgraph/cache.json` (per-file `{hash, entities, edges}`), `<repo>/.docgraph/repos.json` (multi-repo list).

## Kuzu Cypher gotchas (learned the hard way)

- Use `label(r)` for relationship type. **`type(r)` does not exist** in Kuzu.
- `startNode(r)` and `endNode(r)` **do not exist**. Use `(a)-[r]->(b)` and reference `a` / `b` directly.
- `nodes(path)` works for variable-length paths; `relationships(path)` does not on all versions — avoid it.
- Property name on `File` nodes is `path`, not `name`. On all other entity nodes it's `name`. Easy to confuse in `MATCH` clauses.
- `REFERENCES_` (with trailing underscore) — `REFERENCES` is a reserved word in Kuzu.
- Edge tables can declare multiple FROM/TO pairs in one statement (see `db.py::EDGE_DDL`); use that instead of separate edge tables per type-combination.
- Bulk node insert: `UNWIND $rows AS row CREATE (n:Label {col: row.col, ...})` is much faster than per-row CREATE. `db.py::insert_nodes` does this in 5k-row batches.
- Bulk edge insert: `db.py::insert_edges` uses `COPY <Edge> FROM arrow (from='<FromTable>', to='<ToTable>')` (pyarrow-staged Table). The `(from=, to=)` clause is required because rel tables can declare multiple `(FROM, TO)` pairs in one statement. COPY FROM hard-errors on a missing PK — `_known_ids[<table>]` filters dangling rows up front so we keep the old "best-effort, tolerant" semantics.
- A reader connection (`read_only=True`) holds a lock that blocks writers. The MCP server and the web server both open read-only — kill them before running `docgraph index` or you'll get `"Could not set lock on file"`.
- Always call `db.close()` (or let it go out of scope so `__del__` fires) before opening a fresh `read_only=True` connection. On Windows, COPY FROM holds extra internal references that survive a bare `del` — relying on GC alone leaves the file lock held.

## Tree-sitter API gotchas

- We're on `tree-sitter >= 0.25`. The query API changed: use `ts.Query(lang, src)` and `ts.QueryCursor(q).captures(node)` — **not** `lang.query(...).captures(...)` from older docs.
- `captures()` returns `dict[str, list[Node]]`, not the old `list[tuple[Node, str]]`.
- Each language is its own pip package (`tree-sitter-python`, `tree-sitter-typescript`, etc.). The TS package exposes `language_typescript()` and `language_tsx()` (two grammars in one package).
- The `tree-sitter-language-pack` package looks tempting but it's a Rust-native rewrite (`_native` module) that needs runtime grammar downloads from the network. Don't switch to it.

## ID allocation across runs

The indexer uses an integer `id` primary key per entity. On full reindex it starts at 1. On incremental it must continue from `max(id) + 1` in the DB — `_seed_ids_from_db()` handles this. Don't break this invariant; Kuzu rejects duplicate primary keys with a confusing error.

## Per-file delta correctness

The trickiest part of the codebase. Read `index.py::index_all` carefully before changing it. The contract:

1. Hash-compare to bucket files into changed / added / deleted / unchanged.
2. `DETACH DELETE` changed + deleted file nodes — Kuzu removes incident edges in the same step.
3. Re-parse only changed + added files. Update cache.
4. Build the symbol table from current DB state (not from the parse output).
5. For each cached `RawEdge` across all files: insert only if `needs_insert(src_file, target_file)` — i.e., at least one endpoint was just (re)created. Edges fully inside the unchanged set are still in the DB and skipping them prevents duplicates.
6. Tier 4 edges (`SIMILAR_TO`, `CO_CHANGED_WITH`, `TESTS`) and PageRank are wiped and recomputed every time. They're global and cheap.

If you change the parse output shape, update the cache writer **and** the cache reader in lockstep.

## Testing

```bash
.venv/Scripts/python -m pytest                 # ~70s, ~295 tests
```

Tests in `tests/`:
- `test_unit.py` — config, summary/docstring extraction, watch filter, multi-root helpers (no embedder, fast)
- `test_indexer.py` — full + incremental + delete cycles, idempotency, **Variable extraction + CONTAINS + incremental delete**
- `test_retrieval.py` — original 6 retriever methods
- `test_new_tools.py` — `explore`, `impact_of`, `test_impact`, `cypher` (incl. write-blocker tests), personalized PageRank
- `test_multi_repo.py` — multi-root walker, cross-repo indexing, path roundtrip
- `test_cursor_parity.py` — two-tier ignore (`.cursorindexingignore` / `.cursorignore`), `git_*` tools, `rules_for`, sub-function chunking
- `test_round3.py` — cross-encoder reranker (mocked), scope-aware resolution, `@Docs` ingestion (mocked HTTP)
- `test_api.py` — FastAPI HTTP layer (every `/api/*` route, sandboxing, `.cursorignore` redaction, cypher write-blocker via POST)
- `test_mcp_server.py` — every MCP tool registered + invokable end-to-end through `FastMCP.call_tool`
- `test_llm.py` — LLM client unit tests (urllib mocked); covers OpenAI + Anthropic + auth headers + malformed-response fallback
- `test_llm_live.py` — **Optional live integration.** Probes `localhost:1235/v1/models` for `qwen3.6-35b`; auto-skips if either is missing. Override host/port/model via `DOCGRAPH_LLM_TEST_*` env vars.
- `test_daemon.py` — Embedding daemon: ping / embed roundtrip / stale-lock cleanup. Spins up `run_daemon` in a thread on a free port with a sandboxed `LOCK_PATH`, so it never touches the user's real `~/.docgraph/daemon.lock`.
- `test_workspace.py` — `Workspace` registry: 4-step `resolve()`, default-slug fallback, slug listing, `take_writer` / `release_writer` round-trip and the read-only reopen.
- `test_cli_flags.py` — Locks every CLI flag telecode passes (`--workers`, `--gpu`, `--llm-host`, `--llm-port`, `--llm-format`, `--llm-max-tokens`, `--llm-model`). Parametrized so a removed flag fails loudly.
- `test_embed_fallback.py` — GPU→CPU recovery in `Embedder.embed()`: simulates a DirectML `DXGI_ERROR_DEVICE_HUNG` failure mid-inference, verifies the cached session is dropped and a CPU retry succeeds. Also covers "unrelated errors propagate" and "no fallback when already on CPU".

**Kuzu writer-visibility gotcha:** a `kuzu.Connection` opened with `read_only=False` doesn't see its own writes via subsequent `fetch_all` queries in the same process. The conftest fixture (and `test_indexer._index_and_reopen_readonly`) close the writer and reopen read-only after indexing. If you write a new test and reads come back empty, that's the cause.

**Full-reindex swap gotcha:** `Indexer.index_all(incremental=False)` does `self.db.wipe(...)` and **swaps `self.db` with a fresh `GraphDB(...)`** at `index.py:386-387`. The original `writer` reference handed to `Indexer(...)` is stale after a full reindex — close `indexer.db` (the active write-holder), not `writer`, before reopening read-only.

You can also smoke test manually:

```bash
# 1. Clean baseline
rm -rf .docgraph
.venv/Scripts/docgraph index --full
.venv/Scripts/docgraph stats > /tmp/full_stats.txt

# 2. Touch + edit + delete cycle
echo "" >> docgraph/embed.py            # content edit
.venv/Scripts/docgraph index            # should be ~1.3s
git checkout docgraph/embed.py          # revert
.venv/Scripts/docgraph index            # ~1.3s, stats back to baseline
.venv/Scripts/docgraph stats > /tmp/incremental_stats.txt

# 3. Diff — should be identical
diff /tmp/full_stats.txt /tmp/incremental_stats.txt
```

If the diff is non-empty, your change broke incremental correctness.

## MCP tool surface — 15 retriever tools + `list_roots`

Base: `search`, `definition`, `references`, `call_graph`, `file_map`, `neighborhood`.

Differentiators (the wedge against Cursor / Greptile / Sourcegraph):

- `explore(seeds, hops, limit)` — multi-hop BFS from one or more seeds in a single call. Replaces N chained `neighborhood` lookups.
- `impact_of(target, depth)` — blast radius of a file or symbol: callers + importers + co-changed files + tests.
- `test_impact(target)` — tests that exercise this code via `TESTS` edges + reverse `CALLS*`.
- `cypher(query, limit)` — read-only Cypher escape hatch. Rejects writes (CREATE/MERGE/SET/DELETE/...). Lets agents author their own graph queries.
- `git_changes(ref)` — diff-aware retrieval. Working tree / `HEAD` / `main` / SHA. Returns changed files + entities + 1-hop callers — Cursor's `@Commit` joined to the graph in one call.
- `git_blame(file, line_start, line_end?)` — Cursor Blame parity.
- `git_recent(file?, limit)` — recent commits scoped to a file or repo.
- `rules_for(file)` — Cursor-rules ecosystem compatibility: matches `.cursor/rules/*.mdc` by glob plus `AGENTS.md` / `CLAUDE.md` always-on.
- `search_docs(query)` — semantic search across `docgraph docs add <url>`-ingested external docs (Cursor `@Docs` parity).
- `search(rerank=True)` — opt-in cross-encoder rerank over the top 50 candidates. Falls back silently if the model can't load (offline, missing).
- `list_roots()` — discover what roots the host was started with. Returns `[{slug, path, default, watching, last_indexed_at}, ...]`.

**Every retriever-backed tool has a final `root: RootSlug` argument** typed as a dynamic `(str, Enum)` built from the workspace's slugs at host startup. Single-root host → enum has one value, defaulted; LLM doesn't see it. Multi-root host → LLM picks from a closed set; protocol-layer rejection on typos.

Don't add more without a strong reason. `search` accepts `focus_file` / `focus_symbol` for personalized PageRank — prefer threading those through over adding new tools.

## Multi-root architecture

The `Workspace` registry (in `workspace.py`) is the central abstraction. A host process owns one workspace; the workspace owns N `RootSlot` objects, each with its own `Config`, read-only Kuzu connection, and Retriever. The dynamic enum types in `mcp_tools.py` and `server.py` are derived from the workspace's slug list at boot.

**Root resolution.** `Workspace.resolve(value)` does a 4-step lookup:
1. None / "" → default (first registered).
2. Exact match against any registered absolute path.
3. Slug match (case-insensitive, against the lowercased basename).
4. Path-prefix: if `value` is a file path, find the registered root that contains it.

In practice MCP/API callers send slugs (the closed enum forces it). Path-prefix exists for direct Python callers and for the stdio proxy's path-to-slug resolution at startup.

**Indexer-side `extra_roots`** still exists on `Config` — a different concept from workspace multi-root. `extra_roots` lets the indexer walk multiple paths into **one** `.docgraph/graph.kuzu` (for monorepos with sibling projects). The workspace's roots are independent indexes that sit side-by-side; each has its own DB file. Both shapes can coexist: a workspace can register one root whose Config carries `extra_roots`.

## Watch mode

`docgraph host --watch <root>` starts the unified server with a per-root watcher task on the same event loop. Each watcher takes a writer connection from the workspace via `take_writer(root)` for the duration of one reindex pass, runs `Indexer.index_all(incremental=True)` via `asyncio.to_thread`, then releases the writer (which reopens the workspace's read-only handle for that root — Kuzu's writer-visibility quirk). Other roots keep serving from their unaffected RO handles throughout.

A workspace-wide `asyncio.Semaphore(1)` serializes reindexes across roots; otherwise two CPU-bound passes would fight for cores. After each pass, an SSE `reindex_done {repo_slug, ts, events}` event goes out so the live UI refreshes only the affected slot.

The pre-debounce `_WatchFilter` per root drops `node_modules` / `.git` / non-source paths before `awatch` emits them, so a `git checkout` doesn't blow up.

`docgraph watch <path>` (the legacy CLI alias) is now a thin wrapper that builds a single-root workspace and runs `watch_workspace`. `docgraph watch --serve` is equivalent to `docgraph host --watch <path>`.

## Ignore architecture

Three layers, in order of precedence:

1. **Universal baseline** (`ignores.py::UNIVERSAL`) — always-on patterns that match Cursor's built-in defaults: VCS dirs, OS junk, lockfiles, env files, binaries/media, plus unambiguously-named dep/cache dirs (`node_modules/`, `__pycache__/`, `.venv/`, `.next/`, `.gradle/`, `.angular/`, `.tox/`, `.pytest_cache/`, etc.). Applied regardless of project type.
2. **Ecosystem autodetect** (`ignores.py::TEMPLATES` + `_DETECTORS`) — `Config.__post_init__` globs each root for marker files (`package.json`, `pom.xml`, `Cargo.toml`, `*.csproj`, `angular.json`, ...) and unions in the matching template. Templates contain *ambiguously-named* build dirs (`target/`, `build/`, `dist/`, `out/`, `bin/`, `obj/`) that we only want to ignore when the corresponding ecosystem is detected. Detected keys are stored on `cfg.detected_ecosystems[root]`.
3. **User files** — `.gitignore` / `.docgraphignore` / `.cursorindexingignore` (excluded from the index) and `.cursorignore` (indexed but redacted). All layered on top of 1+2.

`Config.is_ignored()` covers tiers 1+2+user-exclude; `Config.is_ai_blocked()` / `ai_blocked_logical()` cover the AI-block tier. Multi-repo callers use `ai_blocked_logical()` to handle prefixed paths.

Adding a new ecosystem: add an entry to `TEMPLATES` and a row to `_DETECTORS` in `ignores.py`. Detection is `Path.glob`-based — `*.csproj` works as a marker.

## Sub-function chunking

`summary.chunk_body(body, language=None)` splits an entity body > 1500 chars into chunks. With a `language` hint it consults `_SCOPE_BOUNDARY_PATTERNS` (regex per language for `def` / `function` / `fn` / `class` / `public` / `@decorator` / etc.) and prefers to cut at scope-boundary lines once the buffer crosses `CHUNK_TARGET_CHARS` (700). When the buffer hits the hard cap (`CHUNK_MAX_CHARS` = 1400) without finding a boundary, it falls back to a mid-body split with overlap to keep cross-chunk context.

Important nuance: scope-aware flushes drop the overlap so the next chunk starts cleanly with the new method/function header. Mid-body (hard-cap) flushes keep overlap. Without a language hint, behavior degrades to the old line-based splitter.

Stored as separate `Chunk` nodes (`parent_qname`, `parent_label`, `file`, `idx`, `embedding`) with `CONTAINS_CHUNK` edges from the parent. `Retriever._chunk_max_sims()` runs the query embedding against ALL chunk vectors once per search call and pools by parent_qname; the score for a parent entity = `max(entity_sim, best_chunk_sim)`. Cheap because there are typically <500 chunks per repo and one matmul handles them all.

Incremental delete: `_delete_files_from_db()` includes a `MATCH (n:Chunk) WHERE n.file IN $files DETACH DELETE n` step before file nodes go.

Adding a language: drop a regex into `_SCOPE_BOUNDARY_PATTERNS` keyed by the same language id used in `parse.py::LANGUAGES`. The pattern is matched against `lstrip()`'d lines, so don't anchor on indentation. Conservative is fine — false negatives just fall through to the line splitter.

## Scope-aware resolution

`index.py` builds `file_imports: dict[str, set[str]]` from cached `IMPORTS` RawEdges before the resolution loop. `resolve()` prefers same-file → imported-file → global candidates. This is a real precision lift over name-only matching but doesn't replace a real LSP — overload resolution within an imported file is still name-based.

## Cross-encoder reranker

`Reranker` class lazy-loads `jinaai/jina-reranker-v1-tiny-en` (~33 MB ONNX) via `fastembed.rerank.cross_encoder.TextCrossEncoder`. Only called when `search(rerank=True)`. Wraps in try/except so an unavailable model degrades gracefully to the bi-encoder ranking — never fails the request.

`RERANK_TOP_K = 50` candidates fed in; results past 50 keep their bi-encoder rank.

## @Docs

- `docs.add_doc(cfg, url)` — fetch URL with stdlib `urllib`, parse HTML via subclassed `HTMLParser` (no BS4 dep), chunk via `chunk_doc()`, embed, store `Doc` rows.
- Idempotent: re-ingesting the same URL deletes prior chunks first.
- `Retriever.search_docs(query, limit)` — pure cosine similarity over `Doc.embedding`. Separate from code search; agent picks which to call.

## LLM-augmented docstrings (opt-in)

- **Off by default.** Activate with `--llm-model <name>` on `docgraph index` (or set `DOCGRAPH_LLM_MODEL=<name>`). Setting just the model is enough — there is no separate `--llm-docstrings` flag; passing the model implies "turn it on." `DOCGRAPH_LLM_DOCSTRINGS=1` still works as an explicit toggle but is rarely needed.
- Talks to a local OpenAI- or Anthropic-compatible server (LM Studio, llama.cpp, vLLM, Ollama). Endpoint defaults: `localhost:1235`, format `openai`.
- Configurable via CLI flags (`--llm-port`, `--llm-format`, `--llm-max-tokens`) or env vars (`DOCGRAPH_LLM_*`). **No settings file** — these are the only knobs.
- **Reasoning models:** every OpenAI-format request carries `reasoning_effort: "none"`. The telecode proxy at port 1235 maps that to per-model "no thinking" knobs (Qwen3: `enable_thinking=false` + `thinking_budget_tokens=0`; DeepSeek-R1: similar). Plain non-reasoning servers ignore the field. Without it, a 150-token budget comes back with empty `content` because reasoning eats it all. If you change the prompt or move to longer outputs, **don't drop this flag**.
- `Indexer._augment_llm_docstrings()` runs after parsing, before embedding. Targets entities of kind `function` / `method` / `class` / `interface` that lack a native docstring. Skips silently if the server is unreachable. Default timeout is 60s per call; concurrency is `min(8, max(2, cfg.workers))`.
- Cache: `.docgraph/llm_docstrings.json` keyed by `sha256(body)`. Survives across runs and across renames (rename-safe). Incrementals only call the LLM for body-changed entities.
- Generated text is read back in `summary.build_embedding_text(..., llm_doc=...)` — used **only** when no native docstring is found, so we never override a real doc.

## GPU embeddings (opt-in)

- **Off by default.** Enable with `--gpu` on `docgraph index` or `DOCGRAPH_GPU=1`.
- Routed through ONNX Runtime, **not torch**. We pass `providers=GPU_PROVIDERS` (CUDA → DirectML → CoreML → ROCm → CPU) into `TextEmbedding(...)` and ORT picks the first one whose package is installed. Users opt in by `pip install onnxruntime-gpu` / `onnxruntime-directml` / `onnxruntime-silicon`. We do **not** depend on those packages — the install stays slim.
- `Embedder._ensure()` wraps the GPU init in try/except: if loading with providers fails (wrong CUDA version, missing DLL, etc.), it logs a warning and reopens the model on CPU. Never fails the index run.
- `cfg.gpu` is forwarded to every `Embedder(...)` site: `Indexer`, `make_app`, `make_mcp`, `watch_repo`, `watch_and_serve`, `docs.add_doc`. So `--gpu` on `index` doesn't help live search unless the same flag is set when launching `serve` / `mcp` — env var `DOCGRAPH_GPU=1` is the cleanest way to make it sticky across processes.
- The reranker (`rerank.py`) doesn't read `cfg.gpu` — fastembed's cross-encoder picks providers from its default. Could be wired through later if the 33 MB Jina model becomes a bottleneck.

## UI engine

Canvas 2D only — Sigma.js / WebGL was removed in 2.1.0 (unreliable on first paint, esm.sh dependency, brittle interaction with the worker physics). The current viewer pairs canvas rendering with a Web Worker running ForceAtlas2-lite on a spatial grid + label-propagation community detection; the worker is bootstrapped from an inline `Blob([WORKER_SRC])` so there's no CDN dependency. Render batches by color and viewport-culls in world coords. No build step.

## Coding conventions

- Type-hint everything. Use `from __future__ import annotations` at the top of files that use `|` union syntax for older Python compat.
- Cypher strings: keep them in `db.py` or `retrieve.py`. Don't scatter them across `mcp_tools.py` / `server.py`.
- Python 3.10 is the floor. Don't use 3.12+ syntax.
- No emojis in code or commit messages unless the user explicitly asks. The user runs Windows + cp1252 console; Unicode in tool descriptions has crashed `await mcp.call_tool` in the past (see commit history for the `∈` incident).
- One short comment max above non-obvious blocks. Don't narrate.

## Performance targets

These should not regress:

| Scenario | Target |
|---|---|
| No-op incremental | < 0.05s |
| Touch (same hash) | < 0.05s |
| 1-file edit | < 2s |
| Full index, 100k LOC | < 10s on a modern laptop |
| MCP tool call | < 200ms typical |

Anything that pushes past these needs a comment explaining why.

## Common dev commands

```bash
.venv/Scripts/pip install -e .                                # editable install
.venv/Scripts/docgraph index --full                           # rebuild from scratch
.venv/Scripts/docgraph host                                   # cwd as the only root → http://127.0.0.1:5500
.venv/Scripts/docgraph host --root /repo-a --root /repo-b     # multi-root
.venv/Scripts/docgraph host --root /repo-a --watch /repo-a    # also reindex on change
.venv/Scripts/docgraph stats                                  # quick health check

# Editor stdio MCP (Cursor / Claude Desktop). Probes the running host.
.venv/Scripts/docgraph mcp /repo-a --transport stdio          # proxy mode if host is up
.venv/Scripts/docgraph mcp /repo-a --transport stdio --standalone  # explicit isolated mode

# Kill any running host holding the DB lock:
taskkill //F //IM python.exe                                  # Windows
pkill -f docgraph                                             # *nix
```

## Things that have broken before — don't repeat

- Putting `∈` or other non-cp1252 chars in MCP tool docstrings → crashes the call on Windows.
- Calling `type(r)` in Cypher → `function TYPE does not exist`.
- The daemon's embed-handler must NOT call `Embedder.embed()` — that wrapper now checks for a running daemon and would route the request right back to itself, causing infinite recursion. Use `embedder._ensure().embed(...)` directly inside `daemon._serve_one`.
- Per-qname method rescoping breaks when two classes have a method of the same name. Fixed by keying the remap on `id(def_node)`. If you re-touch parse.py's "Re-scope methods inside classes" block, keep the node-identity key — string-keying collapses overloads.
- Forgetting that `File.path` is the name property (not `File.name`).
- Re-running the indexer while `docgraph serve` / `docgraph mcp` / `docgraph watch` is running → DB lock error.
- Using `tree-sitter-language-pack 1.6+` thinking it's the old Goldziher API — it isn't, it's a Rust rewrite that needs network downloads.
- Inserting nodes with IDs starting at 1 on an incremental run → duplicate-PK error from Kuzu. Always `_seed_ids_from_db()` first.
- Reading from a Kuzu writer connection right after writing → empty results. Reopen as `read_only=True`.
- After `Indexer.index_all(incremental=False)`, the `GraphDB` you originally passed in is stale — `index_all` swaps `self.db` mid-run. Close `indexer.db`, not the original handle.
- Relying on `del db; gc.collect()` to release the Kuzu file lock on Windows → flaky. Call `db.close()` explicitly. After COPY FROM in particular, internal Kuzu refs survive plain GC.
- Sending a request to a reasoning-model endpoint without `reasoning_effort: "none"` → empty content because the model burned all 150 tokens on `reasoning_content`. Fix is the flag, not bumping max_tokens.
- Using `from __future__ import annotations` in `mcp_tools.py` or `server.py` → Pydantic's lazy `get_type_hints` can't resolve the closure-local `RootSlug` enum and tools/routes blow up at request time with "TypeAdapter is not fully defined". Both files deliberately avoid the future import.
- `str(enum_member)` on a `(str, Enum)` subclass returns `'RootSlug.X'`, not `'x'`. Use `member.value` when looking the slug up in the workspace.
- DirectML embeddings can return `DXGI_ERROR_DEVICE_HUNG` (0x887A0006) mid-inference when another process (e.g. llama.cpp running a 30B+ model with `n_gpu_layers > 0`) is saturating the GPU. The ORT session is **poisoned** after that — every subsequent `model.embed(...)` returns the same Fail. `Embedder.embed()` catches ORT-flavored failures, drops the cached session from `_MODEL_CACHE`, clears `self.providers`, and retries on CPU. Don't remove the recovery wrapper without a replacement; without it `/api/admin/index` 500s on every call once the GPU's been hung.
- Mounting FastMCP into the FastAPI host without setting `lifespan="on"` in the uvicorn config → `/mcp` returns 500 on every request because FastMCP's streamable-HTTP session manager never initializes. The chained lifespan in `make_app` runs both FastAPI startup and FastMCP startup; uvicorn must call it.

## Telecode integration

[Telecode](../.telecode)'s system-tray UI supervises a single `docgraph host` child for all configured roots. The child's MCP tools get bridged into telecode's proxy as managed tools, namespaced as `docgraph_<tool>` (no per-root prefix — agents pass `root=<slug>` per call to scope queries).

Implications:
- **One child, one port, one log.** Telecode spawns `docgraph host --root A --root B [--watch …] --port <N>` and supervises that lifecycle. Adding/removing a root in telecode settings → restart the child. The workspace is immutable for a host's lifetime by design.
- **Don't run `docgraph host` / stdio-mcp manually while telecode owns it** — they'd contend for the configured port. Use `docgraph mcp <path> --transport stdio` (it'll proxy through telecode's running host) or `--standalone` if you really need an isolated process.
- **Auto-start.** `docgraph.host.auto_start: true` in telecode's `settings.json` brings the child up at `main.py:_post_init`.
- **No daemon CLI anymore.** The embedding daemon ran as a separate `docgraph daemon start` process to share embedders across CLIs — pointless once everything lives inside one host. The daemon code is still there but only as an optional internal optimization; no CLI command surfaces it.

No code changes required on the docgraph side — telecode just spawns the existing `host` CLI command. Pointer: `<telecode>/docgraph/` package + the **DocGraph integration** section in `<telecode>/CLAUDE.md` for the full design.

## Known limitations / next-up

- No SCIP / LSP integration → `CALLS` is name-based and will mis-resolve TS / Java overloads. Roadmap.
- LLM-generated docstrings (Greptile-style) require an opt-in API key path — not built yet.
- Embedding model loads fresh per **process** (~1s cold). In-process duplication is solved (see `embed.py::_MODEL_CACHE`); pre-warming a daemon across CLI invocations would help further.
- `IMPORTS_SYMBOL` is now extracted from `@import.symbol` captures in Python (`from x import Y`), JS (`import {Y} from "x"`), TS, and the JS default-import shape. Java's qualified imports (`import a.b.C;`) terminate in the symbol name — the resolver matches the final dotted segment against Class / Function names; no separate tag query needed.
- `OVERRIDES` is derived in `index.py` from `INHERITS` + same-name methods via the inheritance closure (so a grandchild override of a grandparent method is also recorded). Cheap O(classes × methods); runs once per index pass with no embeddings touched.
- The watcher holds a writer lock per root for the duration of each reindex pass; the `Workspace.release_writer()` call after each pass reopens the read-only handle so other roots and the API stay live throughout.
