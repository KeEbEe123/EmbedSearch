from __future__ import annotations

from pathlib import Path
from typing import Any

import apsw
import numpy as np
import sqlite_vec


class ModelMismatchError(Exception):
    pass


def _row_trace(cursor: apsw.Cursor, row: tuple) -> dict[str, Any]:
    desc = cursor.description
    return {desc[i][0]: row[i] for i in range(len(row))}


def _plain_trace(cursor: apsw.Cursor, row: tuple) -> tuple:
    return row


def _connect(db_path: Path) -> apsw.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = apsw.Connection(str(db_path))
    con.enableloadextension(True)
    sqlite_vec.load(con)
    con.enableloadextension(False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _ensure_schema(con: apsw.Connection, dim: int, model_name: str) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            path       TEXT    NOT NULL,
            mtime      REAL    NOT NULL,
            content    TEXT    NOT NULL,
            chunk_idx  INTEGER NOT NULL,
            chunk_text TEXT    NOT NULL
        )
    """)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)"
    )
    con.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_path_chunk "
        "ON documents(path, chunk_idx)"
    )
    con.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    stored = _get_meta(con, "dim")
    if stored is None:
        con.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0("
            f"doc_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        con.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES ('dim', ?), ('model_name', ?)",
            (str(dim), model_name),
        )
    else:
        stored_model = _get_meta(con, "model_name")
        if stored_model != model_name:
            raise ModelMismatchError(
                f"Index was built with '{stored_model}' ({stored}-dim). "
                f"Run 'embedsearch index --force' to rebuild with '{model_name}'."
            )


def get_connection(db_path: Path, dim: int, model_name: str) -> apsw.Connection:
    con = _connect(db_path)
    _ensure_schema(con, dim, model_name)
    return con


def rebuild_index(db_path: Path, dim: int, model_name: str) -> apsw.Connection:
    con = _connect(db_path)
    con.execute("DROP TABLE IF EXISTS embeddings")
    con.execute("DROP TABLE IF EXISTS documents")
    con.execute("DROP TABLE IF EXISTS metadata")
    _ensure_schema(con, dim, model_name)
    return con


def _get_meta(con: apsw.Connection, key: str) -> str | None:
    row = con.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def get_mtime(con: apsw.Connection, path: Path) -> float | None:
    row = con.execute(
        "SELECT mtime FROM documents WHERE path=? LIMIT 1", (str(path),)
    ).fetchone()
    return row[0] if row else None


def delete_by_path(con: apsw.Connection, path: Path) -> None:
    rows = con.execute(
        "SELECT id FROM documents WHERE path=?", (str(path),)
    ).fetchall()
    if rows:
        ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(ids))
        con.execute(f"DELETE FROM embeddings WHERE doc_id IN ({placeholders})", ids)
        con.execute("DELETE FROM documents WHERE path=?", (str(path),))


def insert_chunk(
    con: apsw.Connection,
    path: Path,
    mtime: float,
    content: str,
    chunk_idx: int,
    chunk_text: str,
    embedding: np.ndarray,
) -> None:
    cur = con.execute(
        "INSERT INTO documents(path, mtime, content, chunk_idx, chunk_text) "
        "VALUES (?, ?, ?, ?, ?)",
        (str(path), mtime, content, chunk_idx, chunk_text),
    )
    doc_id = con.last_insert_rowid()
    blob = embedding.astype(np.float32).tobytes()
    con.execute(
        "INSERT INTO embeddings(doc_id, embedding) VALUES (?, ?)", (doc_id, blob)
    )


def get_all_chunks(con: apsw.Connection) -> list[dict]:
    con.setrowtrace(_row_trace)
    rows = con.execute("SELECT id, chunk_text FROM documents").fetchall()
    con.setrowtrace(_plain_trace)
    return list(rows)


def knn_search(
    con: apsw.Connection, query_embedding: np.ndarray, k: int
) -> list[tuple[int, float]]:
    blob = query_embedding.astype(np.float32).tobytes()
    rows = con.execute(
        "SELECT doc_id, distance FROM embeddings "
        "WHERE embedding MATCH ? AND k = ? "
        "ORDER BY distance",
        (blob, k),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_chunks_by_ids(con: apsw.Connection, ids: list[int]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    con.setrowtrace(_row_trace)
    rows = con.execute(
        f"SELECT id, path, chunk_text, content FROM documents WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    con.setrowtrace(_plain_trace)
    return list(rows)
