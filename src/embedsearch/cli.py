from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from embedsearch.config import Config

app = typer.Typer(name="embedsearch", help="Hybrid BM25 + semantic search for local files.", add_completion=False)
console = Console()


def _load_cfg(
    model: Optional[str],
    db_path: Optional[Path],
    alpha: Optional[float],
) -> Config:
    cfg = Config.load()
    updates = {}
    if model:
        updates["model_name"] = model
    if db_path:
        updates["index_path"] = str(db_path)
    if alpha is not None:
        updates["alpha"] = alpha
    if updates:
        cfg = cfg.model_copy(update=updates)
    return cfg


@app.command()
def index(
    path: Path = typer.Argument(..., help="File or directory to index", exists=True),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Embedding model name"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-index unchanged files / rebuild on model change"),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Override database path"),
) -> None:
    """Index files into the vector database."""
    from embedsearch import indexer
    from embedsearch.db import ModelMismatchError

    cfg = _load_cfg(model, db_path, None)
    t0 = time.perf_counter()
    try:
        stats = indexer.index_path(path.resolve(), cfg, force=force)
    except ModelMismatchError as e:
        console.print(f"[red]Model mismatch:[/red] {e}")
        raise typer.Exit(1)

    elapsed = time.perf_counter() - t0
    console.print(
        f"[green]Done.[/green] Indexed [bold]{stats['files_indexed']}[/bold] files, "
        f"[bold]{stats['chunks_created']}[/bold] chunks in [bold]{elapsed:.1f}s[/bold]."
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(10, "--top", "-k", help="Number of results to return"),
    alpha: Optional[float] = typer.Option(None, "--alpha", "-a", help="BM25 weight [0–1], default 0.3"),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Override database path"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Embedding model name"),
    scores: bool = typer.Option(False, "--scores", "-s", help="Show BM25 / cosine score breakdown"),
) -> None:
    """Search indexed files using hybrid BM25 + semantic ranking."""
    from embedsearch import searcher
    from embedsearch.db import ModelMismatchError

    cfg = _load_cfg(model, db_path, alpha)
    t0 = time.perf_counter()
    try:
        results = searcher.hybrid_search(query, cfg, k)
    except ModelMismatchError as e:
        console.print(f"[red]Model mismatch:[/red] {e}")
        raise typer.Exit(1)
    elapsed = time.perf_counter() - t0

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    table = Table(
        title=f'Results for: "{query}"  ({len(results)} hits, {elapsed:.2f}s)',
        show_lines=True,
        highlight=True,
    )
    table.add_column("#", style="dim", width=3, no_wrap=True)
    table.add_column("File", style="cyan", no_wrap=False, max_width=40)
    table.add_column("Excerpt", max_width=60)
    if scores:
        table.add_column("Score", style="yellow", width=8)
        table.add_column("BM25", style="blue", width=8)
        table.add_column("Cos", style="magenta", width=8)

    for i, r in enumerate(results, 1):
        excerpt = r.chunk_text[:220].replace("\n", " ").strip()
        if len(r.chunk_text) > 220:
            excerpt += "…"
        row = [str(i), r.path, excerpt]
        if scores:
            row += [f"{r.score:.4f}", f"{r.bm25_score:.4f}", f"{r.cosine_score:.4f}"]
        table.add_row(*row)

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8765, "--port", "-p", help="Bind port"),
) -> None:
    """Start the optional FastAPI server for HTTP access."""
    import uvicorn
    from embedsearch.api import app as fastapi_app

    console.print(f"[green]Starting EmbedSearch API on http://{host}:{port}[/green]")
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command("config")
def config_cmd(
    show: bool = typer.Option(False, "--show", help="Print current config"),
    set_alpha: Optional[float] = typer.Option(None, "--alpha", help="Set default BM25 weight"),
    set_model: Optional[str] = typer.Option(None, "--model", help="Set default embedding model"),
    set_db: Optional[str] = typer.Option(None, "--db", help="Set default database path"),
) -> None:
    """View or update persistent configuration."""
    cfg = Config.load()
    updates = {}
    if set_alpha is not None:
        updates["alpha"] = set_alpha
    if set_model is not None:
        updates["model_name"] = set_model
    if set_db is not None:
        updates["index_path"] = set_db

    if updates:
        cfg = cfg.model_copy(update=updates)
        Config.save(cfg)
        console.print("[green]Config saved.[/green]")

    if show or not updates:
        console.print_json(cfg.model_dump_json(indent=2))
