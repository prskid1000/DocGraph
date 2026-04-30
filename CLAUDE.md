# DocGraph — Working Notes for Claude

This file captures the things you can't infer from the code in 10 seconds. Read it before making non-trivial changes.

## What this project is

A local code knowledge graph backed by a single embedded Kuzu file. Indexes any repo with tree-sitter, embeds entities with fastembed, and exposes everything via 15 MCP tools and a single-page web UI.

It is the **v2 rewrite** of an older Neo4j + ChromaDB + Streamlit + Vite stack. The old code is preserved at tag `v1-legacy`. Do not resurrect any of those dependencies.

## Hard rules

- **One Python package, one process, one DB file.** No new top-level dirs, no separate frontend builds, no microservices.
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
| `docgraph/parse.py` | tree-sitter wrappers + tags queries per language. |
| `docgraph/index.py` | Parallel pipeline + per-file delta updates. **Most complex file.** |
| `docgraph/summary.py` | Builds the embedding text per entity (extracts docstrings/JSDoc/Rust `///` etc.). |
| `docgraph/db.py` | Kuzu schema + bulk insert helpers. |
| `docgraph/embed.py` | fastembed wrapper (BGE-small, 384-dim). `Embedder(providers=...)` passes ORT providers through; `GPU_PROVIDERS` is the default GPU stack (CUDA → DirectML → CoreML → ROCm → CPU). |
| `docgraph/rank.py` | NetworkX PageRank + `PersonalizedRanker` (query-time personalized PR with cached graph). |
| `docgraph/retrieve.py` | Hybrid retrieval + `explore` / `impact_of` / `test_impact` / `cypher` / `git_*` / `rules_for`. **All Cypher lives here or in `db.py`.** |
| `docgraph/git_tools.py` | `git diff` / `blame` / `log` shell-outs, joined to graph entities. |
| `docgraph/rules.py` | Parses `.cursor/rules/*.mdc` + `AGENTS.md` / `CLAUDE.md`; glob-matches per file. |
| `docgraph/rerank.py` | Lazy cross-encoder (`jinaai/jina-reranker-v1-tiny-en`, ~33 MB); used when `search(rerank=True)`. |
| `docgraph/llm.py` | Tiny urllib-based client for OpenAI- or Anthropic-compatible local servers. Used by `--llm-docstrings` to summarize entities lacking native docs. Off by default. |
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
- Bulk insert: `UNWIND $rows AS row CREATE (n:Label {col: row.col, ...})` is much faster than per-row CREATE. The helpers in `db.py::insert_nodes` / `insert_edges` already do this.
- A reader connection (`read_only=True`) holds a lock that blocks writers. The MCP server and the web server both open read-only — kill them before running `docgraph index` or you'll get `"Could not set lock on file"`.

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
.venv/Scripts/python -m pytest                 # ~26s, 134 tests
```

Tests in `tests/`:
- `test_unit.py` — config, summary/docstring extraction, watch filter, multi-root helpers (no embedder, fast)
- `test_indexer.py` — full + incremental + delete cycles, idempotency
- `test_retrieval.py` — original 6 retriever methods
- `test_new_tools.py` — `explore`, `impact_of`, `test_impact`, `cypher` (incl. write-blocker tests), personalized PageRank
- `test_multi_repo.py` — multi-root walker, cross-repo indexing, path roundtrip
- `test_cursor_parity.py` — two-tier ignore (`.cursorindexingignore` / `.cursorignore`), `git_*` tools, `rules_for`, sub-function chunking
- `test_round3.py` — cross-encoder reranker (mocked), scope-aware resolution, `@Docs` ingestion (mocked HTTP)
- `test_api.py` — FastAPI HTTP layer (every `/api/*` route, sandboxing, `.cursorignore` redaction, cypher write-blocker via POST)
- `test_mcp_server.py` — every MCP tool registered + invokable end-to-end through `FastMCP.call_tool`

**Kuzu writer-visibility gotcha:** a `kuzu.Connection` opened with `read_only=False` doesn't see its own writes via subsequent `fetch_all` queries in the same process. The fixture closes the writer and reopens read-only after indexing. If you write a new test and reads come back empty, that's the cause.

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

`summary.chunk_body()` splits an entity body > 1500 chars into ~700-char overlapping chunks aligned to line boundaries. Stored as separate `Chunk` nodes (`parent_qname`, `parent_label`, `file`, `idx`, `embedding`) with `CONTAINS_CHUNK` edges from the parent.

`Retriever._chunk_max_sims()` runs the query embedding against ALL chunk vectors once per search call and pools by parent_qname. The score for a parent entity = `max(entity_sim, best_chunk_sim)`. Cheap because there are typically <500 chunks per repo and one matmul handles them all.

Incremental delete: `_delete_files_from_db()` includes a `MATCH (n:Chunk) WHERE n.file IN $files DETACH DELETE n` step before file nodes go.

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

- **Off by default.** Enable with `--llm-docstrings` on `docgraph index` or `DOCGRAPH_LLM_DOCSTRINGS=1`.
- Talks to a local OpenAI- or Anthropic-compatible server (LM Studio, llama.cpp, vLLM, Ollama). Defaults: `localhost:1235`, model `local-model`, format `openai`.
- Configurable via CLI flags (`--llm-port`, `--llm-model`, `--llm-format`) or env vars (`DOCGRAPH_LLM_*`). **No settings file** — these are the only knobs.
- `Indexer._augment_llm_docstrings()` runs after parsing, before embedding. Targets entities of kind `function` / `method` / `class` / `interface` that lack a native docstring. Skips silently if the server is unreachable.
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
- Forgetting that `File.path` is the name property (not `File.name`).
- Re-running the indexer while `docgraph serve` / `docgraph mcp` / `docgraph watch` is running → DB lock error.
- Using `tree-sitter-language-pack 1.6+` thinking it's the old Goldziher API — it isn't, it's a Rust rewrite that needs network downloads.
- Inserting nodes with IDs starting at 1 on an incremental run → duplicate-PK error from Kuzu. Always `_seed_ids_from_db()` first.
- Reading from a Kuzu writer connection right after writing → empty results. Reopen as `read_only=True`.

## Known limitations / next-up

- No SCIP / LSP integration → `CALLS` is name-based and will mis-resolve TS / Java overloads. Roadmap.
- LLM-generated docstrings (Greptile-style) require an opt-in API key path — not built yet.
- Embedding model loads fresh per process (~1s cold). Pre-warming or caching across CLI invocations would help.
- `IMPORTS_SYMBOL` and `OVERRIDES` edges declared in the schema but not extracted by any tags query yet.
- The watcher holds a writer lock for its lifetime — kill `serve` / `mcp` first.
