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
console = Console()

LOG_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _parse_ext_csv(raw: str | None) -> tuple[str, ...] | None:
    """Parse a CSV extension list into a normalized tuple. None / empty
    returns None so load_config falls back to its defaults."""
    if not raw:
        return None
    out = tuple(
        e.strip().lstrip(".").lower()
        for e in raw.split(",") if e.strip()
    )
    return out or None


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
    documents: bool = typer.Option(
        False, "--documents/--no-documents",
        help="Also index repo documents (.md/.txt/.rst/small CSVs) and "
             "register heavy / binary files (.pdf/.xlsx/.png/.mp4/etc.) "
             "as Asset nodes with REFERENCES_ edges from any code/doc that "
             "mentions them by path. Off by default.",
    ),
    text_exts: str = typer.Option(
        "", "--text-exts",
        help="Comma-separated list of extensions for the text-doc tier "
             "(default: md,markdown,txt,rst,csv). Implies --documents.",
    ),
    asset_exts: str = typer.Option(
        "", "--asset-exts",
        help="Comma-separated list of extensions for the asset tier "
             "(default: pdf,xlsx,docx,png,jpg,svg,mp4,parquet,zip,...). "
             "Implies --documents.",
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
    text_exts_t = _parse_ext_csv(text_exts)
    asset_exts_t = _parse_ext_csv(asset_exts)
    cfg = load_config(
        path,
        extra_roots=repo if repo else None,
        gpu=gpu,
        embedding_model=embed_model or "BAAI/bge-small-en-v1.5",
        llm_docstrings=bool(llm_model),
        llm_host=llm_host,
        llm_port=llm_port,
        llm_model=llm_model or "qwen3.6-35b",
        llm_format=llm_format,
        llm_max_tokens=llm_max_tokens,
        index_documents=bool(documents or text_exts or asset_exts),
        text_extensions=text_exts_t,
        asset_extensions=asset_exts_t,
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
    stats = indexer.index_all(incremental=not full)
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
                f"[yellow]Root {cfg.repo_root} has no index yet. "
                f"Run `docgraph index {cfg.repo_root}` first.[/yellow]"
            )
            raise typer.Exit(1)
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
        help="Per-call token budget.",
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
    # Document + asset indexing (opt-in, mirrors `docgraph index`).
    documents: bool = typer.Option(
        False, "--documents",
        help="Enable the tier-2/3 document + asset pass on indexes "
             "triggered through this host.",
    ),
    text_exts: str | None = typer.Option(
        None, "--text-exts",
        help="Comma-separated text extensions. Implies --documents. ",
    ),
    asset_exts: str | None = typer.Option(
        None, "--asset-exts",
        help="Comma-separated asset extensions. Implies --documents. ",
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

    overrides: dict = {
        "host": bind_host,
        "port": bind_port,
        "gpu": gpu,
        "rerank_default": rerank_default,
        "rerank_gpu": rerank_gpu,
        "index_documents": bool(documents or text_exts or asset_exts),
    }
    if embed_model:                overrides["embedding_model"] = embed_model
    if embed_dim:                  overrides["embedding_dim"] = embed_dim
    if rerank_model:               overrides["rerank_model"] = rerank_model
    if llm_model:
        overrides["llm_model"] = llm_model
    overrides["llm_docstrings"] = llm_docstrings
    overrides["llm_wiki"] = llm_wiki
    if llm_host:                   overrides["llm_host"] = llm_host
    if llm_port:                   overrides["llm_port"] = llm_port
    if llm_format:                 overrides["llm_format"] = llm_format
    if llm_max_tokens:             overrides["llm_max_tokens"] = llm_max_tokens
    text_exts_t = _parse_ext_csv(text_exts)
    asset_exts_t = _parse_ext_csv(asset_exts)
    if text_exts_t:                overrides["text_extensions"] = text_exts_t
    if asset_exts_t:               overrides["asset_extensions"] = asset_exts_t

    roots = _resolve_roots(path, root)
    workspace = _build_workspace(roots, **overrides)
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
        pages = build_wiki(cfg, db, llm, only_module=module, progress=_progress, force=force, depth=depth)
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


docs_app = typer.Typer(help="Manage external documentation (Cursor @Docs parity).")
app.add_typer(docs_app, name="docs")


@docs_app.command("add")
def docs_add(
    url: str = typer.Argument(..., help="Documentation URL to fetch and index"),
    path: Path = typer.Option(Path.cwd(), "--path", help="Repo whose .docgraph/ to write to"),
) -> None:
    """Fetch a URL, chunk + embed it, store as Doc nodes for semantic search."""
    cfg = load_config(path)
    from docgraph.docs import add_doc
    console.print(f"[cyan]Fetching[/cyan] {url}")
    try:
        out = add_doc(cfg, url)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed:[/red] {e}")
        raise typer.Exit(1)
    if "error" in out:
        console.print(f"[yellow]{out['error']}[/yellow]")
        raise typer.Exit(1)
    console.print(f"[green]Indexed[/green] {out['chunks']} chunks — {out['title'] or url}")


@docs_app.command("list")
def docs_list(
    path: Path = typer.Option(Path.cwd(), "--path"),
) -> None:
    """List all ingested doc URLs and their chunk counts."""
    cfg = load_config(path)
    from docgraph.docs import list_docs
    rows = list_docs(cfg)
    if not rows:
        console.print("[yellow]No docs ingested yet. Try:[/yellow] docgraph docs add <url>")
        return
    table = Table(title="Ingested docs")
    table.add_column("Source"); table.add_column("Title"); table.add_column("Chunks", justify="right")
    for r in rows:
        table.add_row(r["source"], r.get("title") or "", str(r["chunks"]))
    console.print(table)


@docs_app.command("remove")
def docs_remove(
    url: str = typer.Argument(...),
    path: Path = typer.Option(Path.cwd(), "--path"),
) -> None:
    """Delete all chunks for a previously-ingested doc URL."""
    cfg = load_config(path)
    from docgraph.docs import remove_doc
    n = remove_doc(cfg, url)
    console.print(f"[green]Removed[/green] {n} chunks for {url}")


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
