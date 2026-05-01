"""docgraph CLI."""
from __future__ import annotations

import asyncio
import json
import logging
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

app = typer.Typer(
    name="docgraph",
    help="Local code knowledge graph: index any repo, query via MCP or web UI.",
    no_args_is_help=True,
)
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
    llm_port: int = typer.Option(
        1235, "--llm-port", help="Local LLM server port (default: 1235). Ignored unless --llm-model is set.",
    ),
    llm_format: str = typer.Option(
        "openai", "--llm-format",
        help="API format: 'openai' (Chat Completions) or 'anthropic' (Messages). Ignored unless --llm-model is set.",
    ),
    llm_max_tokens: int = typer.Option(
        150, "--llm-max-tokens",
        help="Max tokens per LLM call (default: 150). docgraph sends "
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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Index a codebase. Incremental by default; pass --full to wipe and rebuild.

    Pass `--repo` (repeatable) to index multiple repos into one graph. The list
    is persisted; subsequent commands (watch, serve, mcp) will see all repos
    automatically.

    LLM docstrings (opt-in): pass `--llm-model <name>` to ask a local LLM to
    write a one-sentence summary for each entity that has no native docstring.
    Adjust the endpoint via `--llm-port` / `--llm-format` / `--llm-max-tokens`
    or the matching `DOCGRAPH_LLM_*` env vars. Generated summaries are cached
    by body hash in `.docgraph/llm_docstrings.json` so incrementals don't
    re-call the model.

    GPU (opt-in): with `--gpu`, embeddings run on the GPU via ONNX Runtime.
    No torch dependency — install `onnxruntime-gpu` (NVIDIA / CUDA),
    `onnxruntime-directml` (Windows / any GPU), or `onnxruntime-silicon`
    (Apple Silicon) to light it up. Silently falls back to CPU otherwise.
    """
    _setup_logging(verbose)
    cfg = load_config(path, extra_roots=repo if repo else None)
    # CLI flags win over env vars. Setting --llm-model enables the feature.
    if llm_model is not None:
        cfg.llm_docstrings = True
        cfg.llm_model = llm_model
        cfg.llm_port = llm_port
        cfg.llm_format = llm_format
        cfg.llm_max_tokens = llm_max_tokens
    if gpu:
        cfg.gpu = True
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
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    indexer = Indexer(cfg, db)
    stats = indexer.index_all(incremental=not full)
    table = Table(show_header=False, box=None)
    for k, v in stats.items():
        table.add_row(k, f"{v:.2f}" if isinstance(v, float) else str(v))
    console.print(table)


@app.command()
def watch(
    path: Path = typer.Argument(Path.cwd(), help="Repo root (default: cwd)"),
    debounce: int = typer.Option(500, "--debounce", help="Debounce window (ms) before reindex fires"),
    serve: bool = typer.Option(
        False, "--serve",
        help="Also run the web UI + JSON API in the same process. The UI auto-redraws on each reindex via SSE.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address for --serve mode."),
    port: int = typer.Option(5500, "--port", help="Bind port for --serve mode."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Watch the repo and incrementally reindex on file changes.

    Plain `watch` holds a writer lock — kill any `docgraph serve` / `docgraph mcp`
    against the same repo first.

    `watch --serve` runs the web UI + JSON API in the same process, so they share
    a single DB lock. The browser stays in sync via Server-Sent Events at
    `/api/events`; the graph re-renders automatically after each reindex.
    """
    _setup_logging(verbose)
    cfg = load_config(path)
    if serve:
        from docgraph.watch import watch_and_serve
        asyncio.run(watch_and_serve(cfg, debounce_ms=debounce, host=host, port=port))
        return
    console.print(f"[cyan]Watching[/cyan] {cfg.repo_root}  debounce={debounce}ms")
    from docgraph.watch import watch_repo
    watch_repo(cfg, debounce_ms=debounce)


@app.command()
def serve(
    path: Path = typer.Argument(Path.cwd(), help="Repo root"),
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the web UI + JSON API."""
    _setup_logging(verbose)
    cfg = load_config(path)
    if host:
        cfg.host = host
    if port:
        cfg.port = port
    if not cfg.db_path.exists():
        console.print("[yellow]No index yet. Run `docgraph index` first.[/yellow]")
        raise typer.Exit(1)
    from docgraph.server import make_app
    app_obj = make_app(cfg)
    console.print(f"[green]Serving[/green] http://{cfg.host}:{cfg.port}/")
    uvicorn.run(app_obj, host=cfg.host, port=cfg.port, log_level="info" if verbose else "warning")


@app.command()
def mcp(
    path: Path = typer.Argument(Path.cwd(), help="Repo root"),
    transport: str = typer.Option("stdio", help="stdio | http"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the MCP server (stdio for Cursor/Claude, http for web clients)."""
    _setup_logging(verbose)
    cfg = load_config(path)
    if not cfg.db_path.exists():
        console.print("[yellow]No index yet. Run `docgraph index` first.[/yellow]", file=sys.stderr)
        raise typer.Exit(1)
    from docgraph.mcp_tools import make_mcp
    server = make_mcp(cfg)
    if transport == "stdio":
        server.run()
    elif transport == "http":
        server.run(transport="http", host=cfg.host, port=cfg.port)
    else:
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
