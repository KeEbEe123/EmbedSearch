"""Tests for db.py using an in-memory SQLite database."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest
import sqlite_vec

from embedsearch import db
from embedsearch.db import ModelMismatchError


DIM = 4
MODEL = "test-model"


def _make_con(path: Path) -> sqlite3.Connection:
    return db.get_connection(path, DIM, MODEL)


def _rand_vec() -> np.ndarray:
    v = np.random.rand(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def tmp_db(tmp_path):
    return tmp_path / "test.db"


def test_schema_created(tmp_db):
    con = _make_con(tmp_db)
    tables = {
        r[0]
        for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "documents" in tables
    assert "metadata" in tables


def test_insert_and_retrieve(tmp_db):
    con = _make_con(tmp_db)
    path = Path("/fake/file.py")
    vec = _rand_vec()
    with con:
        db.insert_chunk(con, path, 1234.0, "full content", 0, "chunk text", vec)

    rows = db.get_all_chunks(con)
    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "chunk text"


def test_delete_by_path(tmp_db):
    con = _make_con(tmp_db)
    path = Path("/fake/file.py")
    with con:
        db.insert_chunk(con, path, 1234.0, "content", 0, "chunk", _rand_vec())
        db.insert_chunk(con, path, 1234.0, "content", 1, "chunk2", _rand_vec())
    assert len(db.get_all_chunks(con)) == 2

    with con:
        db.delete_by_path(con, path)
    assert len(db.get_all_chunks(con)) == 0


def test_mtime_detection(tmp_db):
    con = _make_con(tmp_db)
    path = Path("/fake/file.py")
    with con:
        db.insert_chunk(con, path, 9999.0, "c", 0, "t", _rand_vec())
    assert db.get_mtime(con, path) == 9999.0
    assert db.get_mtime(con, Path("/nonexistent")) is None


def test_model_mismatch_raises(tmp_db):
    _make_con(tmp_db)  # creates index with MODEL
    with pytest.raises(ModelMismatchError):
        db.get_connection(tmp_db, DIM, "different-model")


def test_knn_search(tmp_db):
    con = _make_con(tmp_db)
    target = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    with con:
        db.insert_chunk(con, Path("/a.txt"), 1.0, "c", 0, "target chunk", target)
        db.insert_chunk(con, Path("/b.txt"), 1.0, "c", 0, "other chunk", other)

    results = db.knn_search(con, target, 2)
    assert len(results) == 2
    # First result should be the most similar (target itself)
    top_id, top_dist = results[0]
    assert top_dist < 0.01  # near-zero distance for identical vector
