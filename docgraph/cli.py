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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Index a codebase. Incremental by default; pass --full to wipe and rebuild."""
    _setup_logging(verbose)
    cfg = load_config(path)
    console.print(f"[cyan]Indexing[/cyan] {cfg.repo_root}")
    console.print(f"  workers: {cfg.workers}  db: {cfg.db_path}")
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    indexer = Indexer(cfg, db)
    stats = indexer.index_all(incremental=not full)
    table = Table(show_header=False, box=None)
    for k, v in stats.items():
        table.add_row(k, f"{v:.2f}" if isinstance(v, float) else str(v))
    console.print(table)


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
