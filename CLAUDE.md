# DocGraph — Working Notes for Claude

This file captures the things you can't infer from the code in 10 seconds. Read it before making non-trivial changes.

## What this project is

A local code knowledge graph backed by a single embedded Kuzu file. Indexes any repo with tree-sitter, embeds entities with fastembed, and exposes everything via 15 MCP tools and a single-page web UI.

It is the **v2 rewrite** of an older Neo4j + ChromaDB + Streamlit + Vite stack. The old code is preserved at tag `v1-legacy`. Do not resurrect any of those dependencies.

## Hard rules

- **One Python package, one process, one DB file.** No new top-level dirs, no separate frontend builds, no microservices. (The optional embedding daemon at `docgraph daemon start` is a deliberate exception — opt-in, loopback-only, never required for any feature; the codepath always falls back to in-process embedding if the daemon is down.)
- **Kuzu is the only data store.** Don't add SQLite, ChromaDB, Neo4j, Redis, etc.
- **No npm.** The web UI is one HTML file at `docgraph/ui/index.html`. No build step.
- **No torch dependency.** Embeddings go through `fastembed` (ONNX). Adding sentence-transformers/torch would 4× the install size. GPU support uses ONNX Runtime providers (`CUDAExecutionProvider` / `DmlExecutionProvider` / `CoreMLExecutionProvider`) — not torch CUDA. Users opt in by `pip install onnxruntime-gpu` (or `-directml` / `-silicon`); we never depend on those.
- **Per-language processor classes are forbidden.** All language support comes from `parse.py::LANGUAGES` + `TAGS_QUERIES`. Adding a language = pip install `tree-sitter-<x>` and add two dict entries.

## File map

| File | Purpose |
|---|---|
| `docgraph/cli.py` | Typer entry. Add subcommands here. |
| `docgraph/config.py` | Auto-detects repo root + ecosystems, respects `.gitignore` + `.docgraphignore`. Multi-root via `extra_roots`; persisted in `.docgraph/repos.json`. |
| `docgraph/ignores.py` | Universal ignore patterns + per-ecosystem autodetect (Node / Python / Maven / Gradle / Rust / .NET / Swift / Ruby / Dart / Elixir / Scala / PHP / Terraform / Unity / Go). Universal layer also covers Jupyter / MLflow / wandb / DVC / Haskell / Zig / R / Scala tooling. Inline string lists, no template files. |
| `docgraph/parse.py` | tree-sitter wrappers + tags queries per language. Method qname rescoping is keyed on `id(node)` not on the original qname string — methods sharing a base qname (`file::area` for both `Square.area` and `Circle.area`) used to collide and remap to one final qname. |
| `docgraph/daemon.py` | Optional cross-CLI embedding daemon. Length-prefixed JSON on loopback TCP (default port 5577). `is_running()` checks the lock file at `~/.docgraph/daemon.lock` and pings the recorded port — stale locks (PID dead, port silent) are auto-cleared. `Embedder.embed()` consults `embed_via_daemon()` before loading its own ONNX session; daemon-side `_serve_one()` calls `embedder._ensure().embed(...)` directly to avoid recursing back through the wrapper. |
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
| `docgraph/watch.py` | `watchfiles` loop with pre-debounce ignore filter. `watch_and_serve()` runs uvicorn + the watcher in one event loop and broadcasts SSE `reindex_done` on each cycle. |
| `docgraph/mcp_tools.py` | 15 MCP tools (6 base + 9 differentiators). Keep this surface tight. |
| `docgraph/server.py` | FastAPI: web UI + JSON API + SSE `/api/events`. `make_app(cfg, db=None)` accepts a pre-opened DB; `app.state.db_holder` is a swap-safe `(db, retriever)` wrapper used by `watch --serve`. |
| `docgraph/ui/index.html` | Single-page graph viewer. Canvas + Sigma.js (lazy-loaded from esm.sh, auto-engages > 2 k nodes). |

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
.venv/Scripts/python -m pytest                 # ~65s, 178 tests (incl. daemon)
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

## MCP tool surface — 6 base + 9 differentiators

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

Don't add more without a strong reason. `search` accepts `focus_file` / `focus_symbol` for personalized PageRank — prefer threading those through over adding new tools.

## Multi-repo

`extra_roots: list[Path]` on `Config`, persisted in `.docgraph/repos.json`. Walker emits `(absolute_path, logical_rel)` where `logical_rel = "<repo>/<rel>"` in multi-root mode. Cross-repo `IMPORTS` resolve via the existing fuzzy substring match — no schema change. `_write_co_changed` runs `git log` per root and prefixes paths.

## Watch mode

`docgraph watch` opens a writer connection for its lifetime — kill `serve` / `mcp` against the same DB first. Pre-debounce filter (`_WatchFilter`) drops ignored / unsupported-language paths *before* watchfiles emits them so `git checkout` of `node_modules` doesn't fire 5 k events.

`docgraph watch --serve` runs the watcher AND uvicorn in one asyncio loop (`watch.watch_and_serve`):

1. Opens a writer DB → does a baseline incremental reindex → closes the writer.
2. Reopens read-only and starts uvicorn. The API uses `app.state.db_holder` (a `threading.Lock`-guarded `(db, retriever)` pair) so handlers always see a consistent snapshot.
3. On each `awatch` change batch:
   a. Acquire the holder lock; close the read-only DB; open writer; swap into the holder.
   b. Run `Indexer.index_all(incremental=True)` via `asyncio.to_thread()` so SSE keepalives keep flowing.
   c. Acquire lock again; close writer; reopen read-only; swap back. (Required because Kuzu writer connections don't see their own writes via subsequent `fetch_all` queries — the close+reopen forces visibility.)
   d. `server.broadcast(app, "reindex_done", {...})` pushes to every active SSE subscriber. UI re-fetches `/api/graph` + `/api/stats`.

Single-process design avoids the Kuzu file-lock conflict that otherwise blocks `serve` from coexisting with `watch`.

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

## UI engines

Canvas (default, O(N²) force) and Sigma.js (WebGL, lazy-loaded from esm.sh). Auto-engages > 2 k nodes; manual toggle button. No build step.

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
.venv/Scripts/pip install -e .                 # editable install
.venv/Scripts/docgraph index --full            # rebuild from scratch
.venv/Scripts/docgraph serve                   # http://127.0.0.1:5500
.venv/Scripts/docgraph stats                   # quick health check

# Kill any running servers/MCPs holding the DB lock:
taskkill //F //IM python.exe                   # Windows
pkill -f docgraph                              # *nix
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

## Known limitations / next-up

- No SCIP / LSP integration → `CALLS` is name-based and will mis-resolve TS / Java overloads. Roadmap.
- LLM-generated docstrings (Greptile-style) require an opt-in API key path — not built yet.
- Embedding model loads fresh per **process** (~1s cold). In-process duplication is solved (see `embed.py::_MODEL_CACHE`); pre-warming a daemon across CLI invocations would help further.
- `IMPORTS_SYMBOL` is now extracted from `@import.symbol` captures in Python (`from x import Y`), JS (`import {Y} from "x"`), TS, and the JS default-import shape. Java's qualified imports (`import a.b.C;`) terminate in the symbol name — the resolver matches the final dotted segment against Class / Function names; no separate tag query needed.
- `OVERRIDES` is derived in `index.py` from `INHERITS` + same-name methods via the inheritance closure (so a grandchild override of a grandparent method is also recorded). Cheap O(classes × methods); runs once per index pass with no embeddings touched.
- The watcher holds a writer lock for its lifetime — kill `serve` / `mcp` first.
