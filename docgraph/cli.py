"""docgraph CLI."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from docgraph import __version__
from docgraph.config import find_repo_root, load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer


def _install_hard_sigint() -> None:
    """Force-exit on Ctrl+C, bypassing uvicorn's graceful shutdown.

    On Windows, asyncio.add_signal_handler() is unimplemented, so uvicorn
    can't reliably wake its event loop from a SIGINT received while blocked
    in I/O — the user hits Ctrl+C and nothing happens. Even when uvicorn
    *does* notice, it tries to drain in-flight requests, but the SSE stream
    the browser holds open to /api/events never closes, so the drain hangs.

    The fix: install a SIGINT handler that calls os._exit(0) directly. It
    skips Python's normal teardown (atexit, finalizers), which means we
    rely on Kuzu's read-only connection cleanup not being load-bearing —
    and it isn't, since read-only connections don't write to the DB.
    """
    def _handler(signum, frame):  # noqa: ARG001
        os._exit(0)
    try:
        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGBREAK"):
            # Windows-specific Ctrl+Break — also force-exit.
            signal.signal(signal.SIGBREAK, _handler)
    except (ValueError, OSError):
        # Some embedded contexts can't install signal handlers; just skip.
        pass

app = typer.Typer(
    name="docgraph",
    help="Local code knowledge graph: index any repo, query via MCP or web UI.",
    no_args_is_help=True,
)

links_app = typer.Typer(name="links", help="Manage external links per root.", no_args_is_help=True)
app.add_typer(links_app)

repos_app = typer.Typer(name="repos", help="Manage extra local paths per root.", no_args_is_help=True)
app.add_typer(repos_app)

console = Console()

LOG_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FMT,
        stream=sys.stderr,
    )


@app.command()
def version() -> None:
    """Print version."""
    console.print(f"docgraph {__version__}")


# ── docgraph links ─────────────────────────────────────────────────────────


@links_app.command("list")
def links_list(
    path: Path = typer.Argument(Path.cwd(), help="Repo root (default: cwd)"),
) -> None:
    """List configured external links and their fetch status."""
    import time as _time
    from docgraph.links import load_links

    cfg = load_config(path)
    links = load_links(cfg.data_dir)
    if not links:
        console.print("[yellow]No external links configured.[/yellow]")
        return
    t = Table(show_header=True, box=None)
    t.add_column("URL")
    t.add_column("Depth", justify="center")
    t.add_column("TTL (h)", justify="center")
    t.add_column("Last Fetched")
    t.add_column("Pages", justify="center")
    for lk in links:
        if lk.last_fetched:
            delta = int(_time.time() - lk.last_fetched)
            age = f"{delta // 3600}h ago" if delta >= 3600 else f"{delta // 60}m ago"
            stale = " [yellow](stale)[/]" if lk.is_stale() else ""
            fetched = age + stale
        else:
            fetched = "[dim]never[/]"
        t.add_row(lk.url, str(lk.depth), str(lk.ttl_hours), fetched,
                  str(lk.page_count) if lk.page_count is not None else "—")
    console.print(t)


@links_app.command("add")
def links_add(
    url: str = typer.Argument(..., help="URL to crawl"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repo root"),
    depth: int = typer.Option(1, "--depth", "-d", help="Crawl depth (1–5)"),
    ttl: float = typer.Option(24.0, "--ttl", help="Hours before stale (default 24)"),
) -> None:
    """Add or update an external link for a root."""
    from docgraph.links import upsert_link

    cfg = load_config(path)
    upsert_link(cfg.data_dir, url, depth=depth, ttl_hours=ttl)
    console.print(f"[green]Added:[/green] {url}  (depth={depth}, ttl={ttl}h)")


@links_app.command("remove")
def links_remove(
    url: str = typer.Argument(..., help="URL to remove"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repo root"),
) -> None:
    """Remove an external link from a root."""
    from docgraph.links import remove_link

    cfg = load_config(path)
    remove_link(cfg.data_dir, url)
    console.print(f"[yellow]Removed:[/yellow] {url}")


@links_app.command("fetch")
def links_fetch(
    path: Path = typer.Argument(Path.cwd(), help="Repo root (default: cwd)"),
    force: bool = typer.Option(False, "--force", "-f",
                                help="Re-fetch even if still fresh (ignore TTL)"),
    url: str | None = typer.Option(None, "--url", help="Fetch only this URL"),
) -> None:
    """Fetch external links now without re-indexing."""
    from docgraph.fetch import fetch_all

    cfg = load_config(path)
    console.print(f"[cyan]Fetching links for {cfg.repo_root}…[/cyan]")
    results = fetch_all(cfg.data_dir, force=force, only_url=url)
    if not results:
        console.print("[yellow]No links configured or nothing to fetch.[/yellow]")
        return
    for u, count in results.items():
        console.print(f"  [green]{u}[/green] → {count} page(s) saved")


# ── docgraph repos ────────────────────────────────────────────────────────


@repos_app.command("list")
def repos_list(
    path: Path = typer.Argument(Path.cwd(), help="Repo root (default: cwd)"),
) -> None:
    """List extra local paths configured for a root."""
    import json as _json

    cfg = load_config(path)
    repos_file = cfg.data_dir / "repos.json"
    if not repos_file.exists():
        console.print("[yellow]No extra paths configured.[/yellow]")
        return
    try:
        raw = _json.loads(repos_file.read_text(encoding="utf-8"))
    except Exception:
        raw = []
    if not raw:
        console.print("[yellow]No extra paths configured.[/yellow]")
        return
    t = Table(show_header=True, box=None)
    t.add_column("Extra path")
    for p in raw:
        t.add_row(str(p))
    console.print(t)


@repos_app.command("add")
def repos_add(
    extra: str = typer.Argument(..., help="Local path to add"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repo root"),
) -> None:
    """Add an extra local path to a root."""
    import json as _json

    cfg = load_config(path)
    repos_file = cfg.data_dir / "repos.json"
    try:
        raw = _json.loads(repos_file.read_text(encoding="utf-8")) if repos_file.exists() else []
    except Exception:
        raw = []
    resolved = str(Path(extra).resolve())
    if resolved not in raw:
        raw.append(resolved)
        repos_file.write_text(_json.dumps(raw))
    console.print(f"[green]Added:[/green] {resolved}")


@repos_app.command("remove")
def repos_remove(
    extra: str = typer.Argument(..., help="Local path to remove"),
    path: Path = typer.Option(Path.cwd(), "--path", "-p", help="Repo root"),
) -> None:
    """Remove an extra local path from a root."""
    import json as _json

    cfg = load_config(path)
    repos_file = cfg.data_dir / "repos.json"
    try:
        raw = _json.loads(repos_file.read_text(encoding="utf-8")) if repos_file.exists() else []
    except Exception:
        raw = []
    resolved = str(Path(extra).resolve())
    raw = [p for p in raw if p != resolved]
    repos_file.write_text(_json.dumps(raw))
    console.print(f"[yellow]Removed:[/yellow] {resolved}")


@app.command()
def index(
    path: Path = typer.Argument(Path.cwd(), help="Repo root (default: cwd)"),
    full: bool = typer.Option(False, "--full", "-f", help="Force full reindex"),
    repo: list[Path] = typer.Option(
        None, "--repo", "-r",
        help="Additional repo root to include (repeatable). Stored in .docgraph/repos.json.",
    ),
    llm_model: str | None = typer.Option(
        None, "--llm-model",
        help="Enable LLM-augmented docstrings for entities missing one. "
             "Pass the model name your local server expects (e.g. 'qwen3.6-35b', "
             "'local-model'). Setting this turns the feature on; defaults below "
             "(port 1235, openai format) apply unless overridden.",
    ),
    llm_host: str = typer.Option(
        "localhost", "--llm-host", help="Local LLM server host (default: localhost). Ignored unless --llm-model is set.",
    ),
    llm_port: int = typer.Option(
        1235, "--llm-port", help="Local LLM server port (default: 1235). Ignored unless --llm-model is set.",
    ),
    llm_format: str = typer.Option(
        "openai", "--llm-format",
        help="API format: 'openai' (Chat Completions) or 'anthropic' (Messages). Ignored unless --llm-model is set.",
    ),
    llm_max_tokens: int = typer.Option(
        512, "--llm-max-tokens",
        help="Max tokens per LLM call (default: 512). docgraph sends "
             "`reasoning_effort=none` so reasoning models (Qwen3 / "
             "DeepSeek-R1) skip thinking and fit in this budget.",
    ),
    llm_docstrings: bool = typer.Option(
        False, "--llm-docstrings/--no-llm-docstrings",
        help="Use the LLM to generate docstrings during indexing. "
             "Off by default — must be enabled explicitly.",
    ),
    llm_wiki: bool = typer.Option(
        False, "--llm-wiki/--no-llm-wiki",
        help="Use the LLM when building wiki pages. Off by default.",
    ),
    gpu: bool = typer.Option(
        False, "--gpu",
        help="Use GPU for embeddings via ONNX Runtime (CUDA / DirectML / CoreML). "
             "Requires `onnxruntime-gpu` (NVIDIA) or `onnxruntime-directml` (Windows) "
             "to be installed; falls back to CPU if unavailable.",
    ),
    embed_batch_size: int | None = typer.Option(
        None, "--embed-batch-size",
        help="Batch size for embedding (default: 256). Lower it (e.g. 32) if "
             "you hit GPU device-hung errors with --gpu / DirectML.",
    ),
    workers: int = typer.Option(
        0, "--workers",
        help="Override the indexer worker count. 0 = auto (max(2, cpu_count - 1)).",
    ),
    embed_model: str | None = typer.Option(
        None, "--embed-model",
        help="HF id of the embedding model.",
    ),
    llm_prompt_docstring_file: Path | None = typer.Option(
        None, "--llm-prompt-docstring-file",
        help="Path to a custom docstring prompt template "
             "(must keep {kind}/{name}/{language}/{body} placeholders). ",
    ),
    fetch_links: bool = typer.Option(
        True, "--fetch-links/--no-fetch-links",
        help="Fetch stale external links before indexing (default: on). "
             "Skips URLs whose TTL has not expired.",
    ),
    force_fetch: bool = typer.Option(
        False, "--force-fetch-links",
        help="Re-fetch ALL external links regardless of TTL.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Index a codebase. Incremental by default; pass --full to wipe and rebuild.

    Pass `--repo` (repeatable) to index multiple repos into one graph. The list
    is persisted; subsequent commands (watch, serve, mcp) will see all repos
    automatically.

    LLM docstrings (opt-in): pass `--llm-model <name>` to ask a local LLM to
    write a one-sentence summary for each entity that has no native docstring.
    Adjust the endpoint via `--llm-port` / `--llm-format` / `--llm-max-tokens`.
    Generated summaries are cached by body hash in
    `.docgraph/llm_docstrings.json` so incrementals don't re-call the model.

    GPU (opt-in): with `--gpu`, embeddings run on the GPU via ONNX Runtime.
    No torch dependency — install `onnxruntime-gpu` (NVIDIA / CUDA),
    `onnxruntime-directml` (Windows / any GPU), or `onnxruntime-silicon`
    (Apple Silicon) to light it up. Silently falls back to CPU otherwise.
    """
    _setup_logging(verbose)
    cfg = load_config(
        path,
        extra_roots=repo if repo else None,
        gpu=gpu,
        embedding_model=embed_model or "BAAI/bge-small-en-v1.5",
        llm_docstrings=llm_docstrings,
        llm_host=llm_host,
        llm_port=llm_port,
        llm_model=llm_model or "qwen3.6-35b",
        llm_format=llm_format,
        llm_max_tokens=llm_max_tokens,
    )
    if workers > 0:
        cfg.workers = workers
    # Install the docstring-prompt override (if any) before the indexer runs.
    if llm_prompt_docstring_file:
        try:
            text = Path(llm_prompt_docstring_file).read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[yellow]Could not read prompt file: {exc}[/yellow]")
        else:
            from docgraph.llm import set_docstring_prompt
            set_docstring_prompt(text)
    if gpu:
        # DirectML can hang the GPU at the default batch size of 256 on
        # consumer cards. Auto-pick a safer default unless the user overrode it.
        if embed_batch_size is None:
            embed_batch_size = 32
    if embed_batch_size is not None:
        cfg.embed_batch_size = embed_batch_size

    console.print(f"[cyan]Indexing[/cyan] {cfg.repo_root}")
    if cfg.extra_roots:
        for r in cfg.extra_roots:
            console.print(f"  + {r}")
    console.print(f"  workers: {cfg.workers}  db: {cfg.db_path}")
    if cfg.gpu:
        console.print("  [magenta]GPU[/]: ONNX Runtime providers (CUDA/DirectML/CoreML/CPU)")
    if cfg.llm_docstrings:
        console.print(
            f"  [yellow]LLM docstrings[/]: {cfg.llm_format} @ "
            f"{cfg.llm_host}:{cfg.llm_port} (model={cfg.llm_model})"
        )
    db = GraphDB(cfg.db_path, embedding_dim=cfg.embedding_dim)
    db.init_schema()
    indexer = Indexer(cfg, db)
    stats = indexer.index_all(incremental=not full,
                               fetch_links=fetch_links, force_fetch=force_fetch)
    table = Table(show_header=False, box=None)
    for k, v in stats.items():
        table.add_row(k, f"{v:.2f}" if isinstance(v, float) else str(v))
    console.print(table)


def _build_workspace(roots: list[Path], **overrides) -> "Workspace":
    """Resolve roots → list[Config] → Workspace. Errors clearly if any
    root has no index yet. `overrides` is forwarded verbatim to load_config
    so every workspace slot picks up the same CLI-flag values."""
    from docgraph.workspace import Workspace
    configs = []
    for r in roots:
        cfg = load_config(r, **overrides)
        if not cfg.db_path.exists():
            console.print(
                f"[yellow]Root {cfg.repo_root} has no index yet — "
                f"initializing empty DB. Index from the UI or run "
                f"`docgraph index {cfg.repo_root}`.[/yellow]"
            )
        configs.append(cfg)
    return Workspace(configs)


def _resolve_roots(positional: Path | None,
                    extra: list[Path] | None) -> list[Path]:
    """Combine positional + repeated --root flags. Falls back to cwd."""
    out: list[Path] = []
    if positional is not None:
        out.append(positional)
    if extra:
        out.extend(extra)
    if not out:
        out.append(Path.cwd())
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)
    return deduped


@app.command()
def host(
    path: Path | None = typer.Argument(None, help="Single root (sugar for --root <path>)."),
    root: list[Path] = typer.Option(
        None, "--root", "-r",
        help="Repo root to register (repeatable). With multiple roots, "
             "every API/MCP call accepts a `root=<slug>` arg to pick.",
    ),
    watch: list[Path] = typer.Option(
        None, "--watch",
        help="Per-root watcher: pass once per root to enable live reindex. "
             "Each value must match one of the registered --root entries.",
    ),
    bind_host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    bind_port: int = typer.Option(5500, "--port", help="Bind port."),
    debounce: int = typer.Option(500, "--debounce", help="Watcher debounce (ms)."),
    # Embeddings + reranker — same surface as `docgraph index`, applied to
    # the host's per-slot Embedder + Retriever.
    gpu: bool = typer.Option(
        False, "--gpu",
        help="Run embeddings on GPU (CUDA, DirectML, CoreML, ROCm, then CPU). ",
    ),
    embed_model: str | None = typer.Option(
        None, "--embed-model",
        help="HF id of the embedding model (default: BAAI/bge-small-en-v1.5). ",
    ),
    embed_dim: int | None = typer.Option(
        None, "--embed-dim",
        help="Force the embedding dim (rare — auto-derived from the model). ",
    ),
    rerank_default: bool = typer.Option(
        False, "--rerank-default",
        help="Default rerank=True on /api/search + MCP search. ",
    ),
    rerank_model: str | None = typer.Option(
        None, "--rerank-model",
        help="HF id of the cross-encoder reranker. ",
    ),
    rerank_gpu: bool = typer.Option(
        False, "--rerank-gpu",
        help="Run the cross-encoder reranker on GPU. Independent of --gpu. ",
    ),
    workers: int = typer.Option(
        0, "--workers",
        help="Indexer parallelism. 0 = max(2, cpu_count - 1). Used by the "
             "host's /api/admin/index path.",
    ),
    embed_batch_size: int = typer.Option(
        0, "--embed-batch-size",
        help="Texts per ONNX session call. 0 = 256 (CPU sweet spot, 32 is "
             "often better on GPU). Lower if --gpu saturates VRAM.",
    ),
    wiki_depth: int = typer.Option(
        12, "--wiki-depth",
        help="Default depth for /api/wiki/build when the request payload "
             "omits it. 1 = top-level dirs only, 12 = leaf folders.",
    ),
    # LLM augmentation knobs (used by index / wiki paths run via API).
    llm_model: str | None = typer.Option(
        None, "--llm-model",
        help="LLM id used by docstring augmentation and wiki generation. "
             "Setting it does NOT enable either feature on its own — pass "
             "--llm-docstrings and/or --llm-wiki explicitly.",
    ),
    llm_docstrings: bool = typer.Option(
        False, "--llm-docstrings/--no-llm-docstrings",
        help="Use the LLM to generate docstrings during indexing. "
             "Off by default — must be enabled explicitly.",
    ),
    llm_wiki: bool = typer.Option(
        False, "--llm-wiki/--no-llm-wiki",
        help="Use the LLM when building wiki pages. Off by default — wiki "
             "falls back to the fact-sheet renderer unless this is enabled.",
    ),
    llm_host: str | None = typer.Option(
        None, "--llm-host", help="LLM server host.",
    ),
    llm_port: int | None = typer.Option(
        None, "--llm-port", help="LLM server port.",
    ),
    llm_format: str | None = typer.Option(
        None, "--llm-format",
        help="openai | anthropic.",
    ),
    llm_max_tokens: int | None = typer.Option(
        None, "--llm-max-tokens",
        help="Per-call token budget for indexing-time docstring generation.",
    ),
    llm_max_tokens_wiki: int | None = typer.Option(
        None, "--llm-max-tokens-wiki",
        help="Per-call token budget for wiki page generation. Default 4096.",
    ),
    llm_max_tokens_chat: int | None = typer.Option(
        None, "--llm-max-tokens-chat",
        help=(
            "Per-call token budget for the right-panel Chat tab. "
            "0 (default) = unlimited on OpenAI-compatible servers; Anthropic "
            "uses an 8192 fallback when 0 is passed (its API requires a value)."
        ),
    ),
    llm_api_key: str | None = typer.Option(
        None, "--llm-api-key",
        help="API key for the LLM server.",
    ),
    llm_timeout: int | None = typer.Option(
        None, "--llm-timeout",
        help="Per-request timeout (s).",
    ),
    llm_prompt_docstring_file: Path | None = typer.Option(
        None, "--llm-prompt-docstring-file",
        help="Path to a custom docstring prompt template. ",
    ),
    llm_prompt_wiki_file: Path | None = typer.Option(
        None, "--llm-prompt-wiki-file",
        help="Path to a custom wiki output-format tail. ",
    ),
    # Lock manager timeouts. Reads block briefly when a writer is held;
    # writers queue. Tuning these trades 503-bounces for client-side
    # latency. Watcher writes always queue (no timeout).
    read_wait_timeout: float = typer.Option(
        5.0, "--read-wait-timeout",
        help="Seconds a read request waits for a writer to release before "
             "503'ing with Retry-After: 2. Default 5s.",
    ),
    write_wait_timeout: float = typer.Option(
        60.0, "--write-wait-timeout",
        help="Seconds an API index writer waits in queue before "
             "503'ing. Watcher writes ignore this (always queue).",
    ),
    wiki_write_timeout: float = typer.Option(
        180.0, "--wiki-write-timeout",
        help="Wiki builds are longer than incremental indexes; separate "
             "queue timeout. Default 180s.",
    ),
    writer_force_free_after: float = typer.Option(
        300.0, "--writer-force-free-after",
        help="Log + cancel a writer that's held for longer than this. "
             "Doesn't yank the lock (Kuzu mid-COPY would corrupt) — "
             "only flips the cancel token so the holder bails at next "
             "checkpoint.",
    ),
    idle_unload_sec: float = typer.Option(
        0.0, "--idle-unload-sec",
        help="Auto-unload embedding + reranker ONNX sessions after this "
             "many seconds of inactivity. 0 (default) = never unload. "
             "Models reload lazily on the next request.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the unified DocGraph host: web UI + JSON API + MCP HTTP, multi-root.

    Single-root (terminal sugar):
        docgraph host                       # uses cwd
        docgraph host /path/to/repo         # positional
    Multi-root:
        docgraph host --root /repo-a --root /repo-b --watch /repo-a
    """
    _setup_logging(verbose)
    # Install LLM prompt overrides, if any, before building the workspace —
    # the indexer + wiki call paths consult these via the in-process getters.
    if llm_prompt_docstring_file:
        try:
            txt = Path(llm_prompt_docstring_file).read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[yellow]docstring prompt unreadable: {exc}[/yellow]")
        else:
            from docgraph.llm import set_docstring_prompt
            set_docstring_prompt(txt)
    if llm_prompt_wiki_file:
        try:
            txt = Path(llm_prompt_wiki_file).read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[yellow]wiki prompt unreadable: {exc}[/yellow]")
        else:
            from docgraph.wiki import set_wiki_prompt_tail
            set_wiki_prompt_tail(txt)

    if gpu:
        # DirectML can hang at 256; use 32 if we're on GPU.
        if embed_batch_size == 0:
            embed_batch_size = 32

    overrides: dict = {
        "host": bind_host,
        "port": bind_port,
        "gpu": gpu,
        "rerank_default": rerank_default,
        "rerank_gpu": rerank_gpu,
        "wiki_depth": wiki_depth,
    }
    if embed_model:                overrides["embedding_model"] = embed_model
    if embed_dim:                  overrides["embedding_dim"] = embed_dim
    if rerank_model:               overrides["rerank_model"] = rerank_model
    if workers > 0:                overrides["workers"] = workers
    if embed_batch_size > 0:       overrides["embed_batch_size"] = embed_batch_size
    if llm_model:
        overrides["llm_model"] = llm_model
    else:
        overrides["llm_model"] = "qwen3.6-35b"
    overrides["llm_docstrings"] = llm_docstrings
    overrides["llm_wiki"] = llm_wiki
    if llm_host:                   overrides["llm_host"] = llm_host
    if llm_port:                   overrides["llm_port"] = llm_port
    if llm_format:                 overrides["llm_format"] = llm_format
    if llm_max_tokens:             overrides["llm_max_tokens"] = llm_max_tokens
    if llm_max_tokens_wiki:        overrides["llm_max_tokens_wiki"] = llm_max_tokens_wiki
    if llm_max_tokens_chat is not None: overrides["llm_max_tokens_chat"] = llm_max_tokens_chat
    if llm_api_key:                overrides["llm_api_key"] = llm_api_key
    if llm_timeout:                overrides["llm_timeout"] = llm_timeout
    if idle_unload_sec > 0:        overrides["unload_after"] = idle_unload_sec
    roots = _resolve_roots(path, root)
    workspace = _build_workspace(roots, **overrides)
    # Propagate to the workspace so the lifespan can start the unloader.
    workspace.unload_after = float(idle_unload_sec or 0.0)
    # Apply lock-timeout overrides directly on the workspace's LockTimeouts
    # struct (workspace built before this point so it already has defaults
    # via LockTimeouts()).
    from docgraph.locks import LockTimeouts
    workspace.lock_timeouts = LockTimeouts(
        read_wait=read_wait_timeout,
        write_wait=write_wait_timeout,
        wiki_write=wiki_write_timeout,
        force_free_after=writer_force_free_after,
    )
    watch_paths = [p.resolve() for p in (watch or [])]
    # Validate that every --watch points at a registered root.
    registered = set(workspace.roots())
    for w in watch_paths:
        if w not in registered:
            console.print(
                f"[red]--watch {w} is not registered as a --root[/red]"
            )
            raise typer.Exit(2)

    from docgraph.server import make_app
    app_obj = make_app(workspace)
    _install_hard_sigint()

    if watch_paths:
        # Run watchers + uvicorn on the same event loop.
        from docgraph.watch import watch_and_serve_workspace
        try:
            asyncio.run(watch_and_serve_workspace(
                workspace, app_obj, watch_paths,
                host=bind_host, port=bind_port,
                debounce_ms=debounce, verbose=verbose,
            ))
        except KeyboardInterrupt:
            pass
        finally:
            workspace.close()
        return

    console.print(
        f"[green]DocGraph host[/green] http://{bind_host}:{bind_port}/  "
        f"[dim]({len(roots)} root{'s' if len(roots) > 1 else ''})[/]"
    )
    try:
        uvicorn.run(
            app_obj, host=bind_host, port=bind_port,
            log_level="info" if verbose else "warning",
            timeout_graceful_shutdown=1,
        )
    except KeyboardInterrupt:
        pass
    finally:
        workspace.close()


@app.command()
def watch(
    path: Path | None = typer.Argument(None, help="Single root (default: cwd)."),
    root: list[Path] = typer.Option(None, "--root", "-r", help="Watch root (repeatable)."),
    debounce: int = typer.Option(500, "--debounce"),
    serve: bool = typer.Option(False, "--serve", help="Also expose web UI + JSON API + MCP."),
    bind_host: str = typer.Option("127.0.0.1", "--host"),
    bind_port: int = typer.Option(5500, "--port"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Watch one or more roots and incrementally reindex on changes.

    `watch --serve` is equivalent to `host --watch <each-root>`.
    Plain `watch` (no --serve) is the foreground-only variant.
    """
    _setup_logging(verbose)
    roots = _resolve_roots(path, root)
    if serve:
        # Delegate to the host command — same code path.
        host(
            path=None, root=roots, watch=roots,
            bind_host=bind_host, bind_port=bind_port,
            debounce=debounce, verbose=verbose,
        )
        return
    workspace = _build_workspace(roots)
    _install_hard_sigint()
    try:
        from docgraph.watch import watch_workspace
        asyncio.run(watch_workspace(workspace, list(workspace.roots()), debounce_ms=debounce))
    except KeyboardInterrupt:
        pass
    finally:
        workspace.close()


@app.command()
def serve(
    path: Path | None = typer.Argument(None, help="Single root."),
    root: list[Path] = typer.Option(None, "--root", "-r", help="Repeatable."),
    bind_host: str | None = typer.Option(None, "--host"),
    bind_port: int | None = typer.Option(None, "--port"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Thin alias: equivalent to `docgraph host` with no watchers."""
    host(
        path=path, root=root, watch=None,
        bind_host=bind_host or "127.0.0.1",
        bind_port=bind_port or 5500,
        debounce=500, verbose=verbose,
    )


@app.command()
def mcp(
    path: Path | None = typer.Argument(None, help="Single root."),
    root: list[Path] = typer.Option(None, "--root", "-r", help="Repeatable."),
    transport: str = typer.Option("stdio", help="stdio | http"),
    bind_host: str = typer.Option("127.0.0.1", "--host"),
    bind_port: int = typer.Option(5500, "--port"),
    host_url: str | None = typer.Option(
        None, "--host-url",
        help="(stdio only) HTTP host to proxy through. If a host responds at "
             "this URL, stdio acts as a thin proxy. Default: probe http://127.0.0.1:5500.",
    ),
    standalone: bool = typer.Option(
        False, "--standalone",
        help="(stdio only) Skip the host probe and always run a single-process "
             "stdio server. Useful when you don't want stdio to share state "
             "with a running host.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the MCP server.

    stdio (default): for Cursor / Claude Desktop / editors. If a host is
        already running on `--host-url` (default http://127.0.0.1:5500),
        this acts as a thin stdio↔HTTP proxy scoped to the supplied path.
        Otherwise it runs a single-process stdio server. Use `--standalone`
        to skip the probe.
    http: starts an MCP HTTP server scoped to the registered roots. Prefer
        `docgraph host` instead — it bundles MCP HTTP with the web UI.
    """
    _setup_logging(verbose)
    roots = _resolve_roots(path, root)
    _install_hard_sigint()

    if transport == "stdio":
        if standalone:
            # Explicit opt-out: run a self-contained stdio server. The user
            # is on the hook for not double-loading workspaces if a host
            # is also running.
            workspace = _build_workspace(roots)
            from docgraph.mcp_tools import make_mcp
            server = make_mcp(workspace)
            try:
                server.run()
            except KeyboardInterrupt:
                pass
            finally:
                workspace.close()
            return

        # Strict: must talk to a running host.
        probe = host_url or f"http://{bind_host}:{bind_port}"
        from docgraph.mcp_stdio_proxy import run_stdio_proxy
        if run_stdio_proxy(probe, scope_root=roots[0]):
            return
        console.print(
            f"[red]No docgraph host responded at {probe}.[/red]\n"
            f"Start one (e.g. `docgraph host --root {roots[0]}`) and retry, "
            f"or pass `--standalone` to run an isolated stdio server.",
            file=sys.stderr,
        )
        raise typer.Exit(3)

    if transport == "http":
        workspace = _build_workspace(roots)
        from docgraph.mcp_tools import make_mcp
        server = make_mcp(workspace)
        try:
            server.run(transport="http", host=bind_host, port=bind_port)
        except KeyboardInterrupt:
            pass
        finally:
            workspace.close()
        return

    console.print(f"[red]Unknown transport {transport}[/red]", file=sys.stderr)
    raise typer.Exit(2)


@app.command()
def stats(
    path: Path = typer.Argument(Path.cwd()),
) -> None:
    """Print index statistics."""
    cfg = load_config(path)
    if not cfg.db_path.exists():
        console.print("[yellow]No index yet.[/yellow]")
        raise typer.Exit(1)
    db = GraphDB(cfg.db_path, read_only=True)
    table = Table(title=f"DocGraph stats — {cfg.repo_root}")
    table.add_column("Entity")
    table.add_column("Count", justify="right")
    for label in ("File", "Function", "Class", "Variable", "Module"):
        rows = db.fetch_all(f"MATCH (n:{label}) RETURN count(n) AS c")
        table.add_row(label, str(rows[0]["c"]) if rows else "0")
    console.print(table)
    table2 = Table(title="Edges")
    table2.add_column("Type"); table2.add_column("Count", justify="right")
    for edge in ("CONTAINS", "CALLS", "INSTANTIATES", "REFERENCES_", "INHERITS",
                 "IMPLEMENTS", "OVERRIDES", "DECORATED_BY", "IMPORTS",
                 "IMPORTS_SYMBOL", "SIMILAR_TO", "CO_CHANGED_WITH", "TESTS"):
        try:
            rows = db.fetch_all(f"MATCH ()-[r:{edge}]->() RETURN count(r) AS c")
            table2.add_row(edge, str(rows[0]["c"]) if rows else "0")
        except Exception:
            pass
    console.print(table2)


@app.command()
def wiki(
    path: Path = typer.Argument(Path.cwd()),
    module: str | None = typer.Option(None, "--module", "-m", help="Build only this module."),
    llm_host: str = typer.Option(
        "localhost", "--llm-host", help="Host running the local LLM server (default: localhost). Same flag as `index`."
    ),
    llm_port: int = typer.Option(
        1235, "--llm-port", help="Local LLM server port (default: 1235). Same flag as `index`."
    ),
    llm_model: str = typer.Option(
        "qwen3.6-35b", "--llm-model",
        help="Model name to send to the LLM server (e.g. 'qwen3.6-35b', 'local-model'). Same flag as `index`.",
    ),
    llm_format: str = typer.Option(
        "openai", "--llm-format",
        help="API format: 'openai' or 'anthropic'. Same flag as `index`.",
    ),
    llm_max_tokens: int = typer.Option(
        4096, "--llm-max-tokens",
        help="Max tokens per LLM call (default 4096, vs 150 for `index` "
             "docstrings). Generous so deeply-nested module pages have room.",
    ),
    depth: int = typer.Option(
        12, "--depth", "-d",
        help="Max directory levels to bucket files by. 1 = top-level only "
             "(old behavior); 12 (default) = one page per leaf folder for "
             "any reasonable repo. Ignored folders (node_modules/, .venv/, "
             "ecosystem build dirs) are inherited from index time.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Rebuild every page from scratch. Default behavior is resumable: "
             "modules whose page already exists on disk are skipped (no LLM call).",
    ),
    llm_prompt_wiki_file: Path | None = typer.Option(
        None, "--llm-prompt-wiki-file",
        help="Path to a custom wiki output-format tail. Replaces the built-in "
             "tail; the rendered fact block above it stays. ",
    ),
    fetch_links: bool = typer.Option(
        True, "--fetch-links/--no-fetch-links",
        help="Fetch stale external links before building the wiki (default: on).",
    ),
    force_fetch: bool = typer.Option(
        False, "--force-fetch-links",
        help="Re-fetch ALL external links regardless of TTL.",
    ),
) -> None:
    """Generate (or rebuild) the LLM-grounded wiki for the indexed repo.

    Pulls a fact sheet for each top-level module from Kuzu and asks a local
    LLM to write a 200-word page. Falls back to a plain rendering of the
    facts when no LLM is reachable, so the wiki is never blank.

    Uses the same LLM config as `docgraph index --llm-model`.
    """
    from docgraph.wiki import build_wiki, set_wiki_prompt_tail
    from docgraph.llm import LLMClient, LLMConfig

    if llm_prompt_wiki_file:
        try:
            txt = Path(llm_prompt_wiki_file).read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[yellow]wiki prompt unreadable: {exc}[/yellow]")
        else:
            set_wiki_prompt_tail(txt)
    cfg = load_config(path)
    if not cfg.db_path.exists():
        console.print("[yellow]No index yet — run `docgraph index` first.[/yellow]")
        raise typer.Exit(1)
    db = GraphDB(cfg.db_path, read_only=True)
    base = LLMConfig(
        host=llm_host, port=llm_port, model=llm_model,
        format=llm_format, max_tokens=llm_max_tokens,
    )
    llm = LLMClient(base)
    console.print(f"[cyan]Building wiki for {cfg.repo_root}…[/cyan]")
    console.print(f"  LLM: {base.format} @ {base.host}:{base.port} (model={base.model})")

    from docgraph.index import _bar
    bar = _bar()
    task_id: int | None = None

    def _progress(i: int, total: int, mod: str) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = bar.add_task("[cyan]wiki", total=total)
        bar.update(task_id, completed=i, description=f"[cyan]wiki[/] {mod}")

    with bar:
        pages = build_wiki(cfg, db, llm, only_module=module, progress=_progress,
                           force=force, depth=depth,
                           fetch_links=fetch_links, force_fetch=force_fetch)
        if task_id is not None:
            bar.update(task_id, completed=len(pages) or bar.tasks[task_id].total)
    console.print(f"[green]Built {len(pages)} wiki page(s).[/green]")
    console.print(f"  Files at: {cfg.data_dir / 'wiki'}")


@app.command()
def clear(
    path: Path = typer.Argument(Path.cwd()),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete the index for this repo."""
    cfg = load_config(path)
    if not yes:
        confirm = typer.confirm(f"Delete {cfg.data_dir}?")
        if not confirm:
            raise typer.Exit(1)
    import shutil
    if cfg.data_dir.exists():
        shutil.rmtree(cfg.data_dir)
    console.print(f"[green]Cleared[/green] {cfg.data_dir}")


daemon_app = typer.Typer(help="Manage the optional embedding daemon (faster cold start across CLI invocations).")
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start(
    port: int = typer.Option(5577, "--port", help="Loopback TCP port the daemon binds to."),
    model: str = typer.Option(
        "BAAI/bge-small-en-v1.5", "--model",
        help="Embedding model name (must match what your repos were indexed with).",
    ),
    gpu: bool = typer.Option(
        False, "--gpu",
        help="Load the embedding model on GPU via ONNX Runtime providers (CUDA / DirectML / CoreML). "
             "Requires `onnxruntime-gpu` / `onnxruntime-directml` / `onnxruntime-silicon` installed.",
    ),
    detach: bool = typer.Option(
        False, "--detach", "-d",
        help="Spawn the daemon in a background process and return. Default is to run in the foreground.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the embedding daemon. Other docgraph processes on this host
    will route their embed calls through it, sharing one warm ONNX session.

    Foreground (default): blocks. Ctrl+C exits cleanly. Useful for tmux/screen.
    `--detach`: spawns a background process (Windows: DETACHED_PROCESS;
    POSIX: double-fork) and returns. Lock file at ~/.docgraph/daemon.lock.
    """
    _setup_logging(verbose)
    from docgraph.daemon import is_running, run_daemon, LOCK_PATH

    info = is_running()
    if info:
        console.print(f"[yellow]Daemon already running[/]: pid={info.get('pid')} port={info.get('port')}")
        raise typer.Exit(0)

    if detach:
        # Re-launch *this* command with --detach removed so the child runs
        # in the foreground; the parent returns immediately.
        import subprocess
        cmd = [sys.executable, "-m", "docgraph.cli", "daemon", "start",
               "--port", str(port), "--model", model]
        if gpu:
            cmd.append("--gpu")
        if sys.platform.startswith("win"):
            DETACHED = 0x00000008
            CREATE_NEW_GROUP = 0x00000200
            subprocess.Popen(cmd, creationflags=DETACHED | CREATE_NEW_GROUP, close_fds=True)
        else:
            subprocess.Popen(cmd, start_new_session=True, close_fds=True)
        # Wait briefly for the lock file to appear so users see status
        # without re-running.
        import time
        for _ in range(50):
            if LOCK_PATH.exists():
                break
            time.sleep(0.1)
        info2 = is_running()
        if info2:
            console.print(f"[green]Daemon started[/] (detached): pid={info2.get('pid')} port={info2.get('port')}")
        else:
            console.print("[yellow]Daemon launched but lock not visible yet — check `docgraph daemon status`[/]")
        return

    console.print(f"[cyan]Starting daemon[/] on 127.0.0.1:{port} (model={model}, gpu={gpu})  [dim](Ctrl+C to stop)[/]")
    _install_hard_sigint()
    try:
        rc = run_daemon(port=port, model_name=model, gpu=gpu)
    except KeyboardInterrupt:
        rc = 0
    raise typer.Exit(rc)


@daemon_app.command("stop")
def daemon_stop() -> None:
    """Stop the running daemon."""
    from docgraph.daemon import stop_daemon, is_running
    if not is_running():
        console.print("[dim]No daemon running.[/]")
        raise typer.Exit(0)
    ok = stop_daemon()
    if ok:
        console.print("[green]Daemon stopped.[/]")
    else:
        console.print("[yellow]Stop request sent; lock file cleared.[/]")


@daemon_app.command("status")
def daemon_status() -> None:
    """Print whether the daemon is running and its config."""
    from docgraph.daemon import is_running
    info = is_running()
    if not info:
        console.print("[dim]Daemon: not running.[/]")
        raise typer.Exit(1)
    table = Table(show_header=False, box=None)
    for k in ("pid", "host", "port", "model", "gpu", "started"):
        if k in info:
            table.add_row(k, str(info[k]))
    console.print("[green]Daemon: running[/]")
    console.print(table)


@app.command(name="install-mcp")
def install_mcp(
    path: Path = typer.Argument(Path.cwd()),
) -> None:
    """Print the JSON snippet to register DocGraph as an MCP server in Cursor/Claude."""
    repo = find_repo_root(path)
    snippet = {
        "mcpServers": {
            f"docgraph-{repo.name}": {
                "command": "docgraph",
                "args": ["mcp", str(repo)],
            }
        }
    }
    console.print("[cyan]Add this to your MCP client config:[/cyan]")
    console.print_json(json.dumps(snippet))


if __name__ == "__main__":
    app()
