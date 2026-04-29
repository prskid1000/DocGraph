# DocGraph — Working Notes for Claude

This file captures the things you can't infer from the code in 10 seconds. Read it before making non-trivial changes.

## What this project is

A local code knowledge graph backed by a single embedded Kuzu file. Indexes any repo with tree-sitter, embeds entities with fastembed, and exposes everything via 6 MCP tools and a single-page web UI.

It is the **v2 rewrite** of an older Neo4j + ChromaDB + Streamlit + Vite stack. The old code is preserved at tag `v1-legacy`. Do not resurrect any of those dependencies.

## Hard rules

- **One Python package, one process, one DB file.** No new top-level dirs, no separate frontend builds, no microservices.
- **Kuzu is the only data store.** Don't add SQLite, ChromaDB, Neo4j, Redis, etc.
- **No npm.** The web UI is one HTML file at `docgraph/ui/index.html`. No build step.
- **No torch dependency.** Embeddings go through `fastembed` (ONNX). Adding sentence-transformers/torch would 4× the install size.
- **Per-language processor classes are forbidden.** All language support comes from `parse.py::LANGUAGES` + `TAGS_QUERIES`. Adding a language = pip install `tree-sitter-<x>` and add two dict entries.

## File map

| File | Purpose |
|---|---|
| `docgraph/cli.py` | Typer entry. Add subcommands here. |
| `docgraph/config.py` | Auto-detects repo root, respects `.gitignore` + `.docgraphignore`. Multi-root via `extra_roots`; persisted in `.docgraph/repos.json`. |
| `docgraph/parse.py` | tree-sitter wrappers + tags queries per language. |
| `docgraph/index.py` | Parallel pipeline + per-file delta updates. **Most complex file.** |
| `docgraph/summary.py` | Builds the embedding text per entity (extracts docstrings/JSDoc/Rust `///` etc.). |
| `docgraph/db.py` | Kuzu schema + bulk insert helpers. |
| `docgraph/embed.py` | fastembed wrapper (BGE-small, 384-dim). |
| `docgraph/rank.py` | NetworkX PageRank + `PersonalizedRanker` (query-time personalized PR with cached graph). |
| `docgraph/retrieve.py` | Hybrid retrieval + `explore` / `impact_of` / `test_impact` / `cypher` / `git_*` / `rules_for`. **All Cypher lives here or in `db.py`.** |
| `docgraph/git_tools.py` | `git diff` / `blame` / `log` shell-outs, joined to graph entities. |
| `docgraph/rules.py` | Parses `.cursor/rules/*.mdc` + `AGENTS.md` / `CLAUDE.md`; glob-matches per file. |
| `docgraph/rerank.py` | Lazy cross-encoder (`jinaai/jina-reranker-v1-tiny-en`, ~33 MB); used when `search(rerank=True)`. |
| `docgraph/docs.py` | URL fetch + HTML→text + chunking + Doc node ingestion. Cursor `@Docs` parity. |
| `docgraph/watch.py` | `watchfiles` loop with pre-debounce ignore filter. |
| `docgraph/mcp_tools.py` | 14 MCP tools (6 base + 8 differentiators). Keep this surface tight. |
| `docgraph/server.py` | FastAPI: web UI + JSON API. |
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
.venv/Scripts/python -m pytest                 # ~17s, 60 tests
```

Tests in `tests/`:
- `test_unit.py` — config, summary/docstring extraction, watch filter, multi-root helpers (no embedder, fast)
- `test_indexer.py` — full + incremental + delete cycles, idempotency
- `test_retrieval.py` — original 6 retriever methods
- `test_new_tools.py` — `explore`, `impact_of`, `test_impact`, `cypher` (incl. write-blocker tests), personalized PageRank
- `test_multi_repo.py` — multi-root walker, cross-repo indexing, path roundtrip

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

## MCP tool surface — 6 base + 4 differentiators

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

## Two-tier ignore

Cursor parity:
- `.cursorindexingignore` (or `.gitignore` / `.docgraphignore`) → file is **excluded from the index entirely**. The graph never sees it.
- `.cursorignore` → file is **indexed but redacted**. Graph node still exists; `search`/`definition`/`api/file_content` mask `body`/`snippet`/content with `[redacted by .cursorignore]` so the agent can know "this exists" without reading it.

`Config.is_ignored()` checks tier 1; `Config.is_ai_blocked()` and `Config.ai_blocked_logical()` check tier 2. Multi-repo callers use `ai_blocked_logical()` to handle prefixed paths.

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
