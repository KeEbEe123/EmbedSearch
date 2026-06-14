from __future__ import annotations

from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from embedsearch import chunker, db, embedder
from embedsearch.config import Config
from embedsearch.db import ModelMismatchError


def index_path(
    target: Path,
    cfg: Config,
    force: bool = False,
) -> dict:
    dim = embedder.get_dim(cfg.model_name)

    try:
        con = db.get_connection(cfg.db_path(), dim, cfg.model_name)
    except ModelMismatchError:
        if not force:
            raise
        con = db.rebuild_index(cfg.db_path(), dim, cfg.model_name)

    files = chunker.collect_files(target)
    files_indexed = 0
    chunks_created = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing files...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"[cyan]{file_path.name}[/cyan]")
            mtime = file_path.stat().st_mtime
            existing_mtime = db.get_mtime(con, file_path)

            if existing_mtime == mtime and not force:
                progress.advance(task)
                continue

            db.delete_by_path(con, file_path)
            file_chunks = chunker.index_file(file_path, cfg)

            if not file_chunks:
                progress.advance(task)
                continue

            texts = [c.chunk_text for c in file_chunks]
            embeddings = embedder.embed_texts(texts, cfg.model_name)

            with con:
                for chunk, vec in zip(file_chunks, embeddings):
                    db.insert_chunk(
                        con,
                        file_path,
                        mtime,
                        chunk.content,
                        chunk.chunk_idx,
                        chunk.chunk_text,
                        vec,
                    )

            files_indexed += 1
            chunks_created += len(file_chunks)
            progress.advance(task)

    return {"files_indexed": files_indexed, "chunks_created": chunks_created}
