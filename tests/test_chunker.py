"""Tests for chunker.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from embedsearch.chunker import chunk_text, collect_files, index_file
from embedsearch.config import Config


@pytest.fixture()
def cfg():
    return Config(chunk_size=50, chunk_overlap=10)


def test_chunk_text_small():
    chunks = chunk_text("hello world", 512, 64)
    assert chunks == ["hello world"]


def test_chunk_text_splits():
    # Generate text that definitely exceeds 50 tokens
    long_text = " ".join([f"word{i}" for i in range(200)])
    chunks = chunk_text(long_text, 50, 10)
    assert len(chunks) > 1
    # Each chunk should be non-empty
    for c in chunks:
        assert c.strip()


def test_chunk_text_overlap():
    long_text = " ".join([f"word{i}" for i in range(300)])
    chunks = chunk_text(long_text, 50, 25)
    # With overlap the total token coverage > first chunk alone
    assert len(chunks) >= 2


def test_index_file_txt(tmp_path, cfg):
    f = tmp_path / "test.txt"
    f.write_text("Hello world\nThis is a test file.")
    records = index_file(f, cfg)
    assert len(records) >= 1
    assert records[0].chunk_text
    assert records[0].path == f


def test_index_file_json(tmp_path, cfg):
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"key": "value", "list": [1, 2, 3]}))
    records = index_file(f, cfg)
    assert len(records) >= 1
    assert "key" in records[0].chunk_text


def test_index_file_unsupported(tmp_path, cfg):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG\r\n")
    records = index_file(f, cfg)
    assert records == []


def test_collect_files_directory(tmp_path):
    (tmp_path / "a.py").write_text("print('hello')")
    (tmp_path / "b.md").write_text("# Docs")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "d.ts").write_text("const x = 1;")

    files = collect_files(tmp_path)
    names = {f.name for f in files}
    assert "a.py" in names
    assert "b.md" in names
    assert "d.ts" in names
    assert "c.bin" not in names


def test_collect_files_skips_hidden(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]")
    (tmp_path / "real.py").write_text("x = 1")

    files = collect_files(tmp_path)
    names = {f.name for f in files}
    assert "config" not in names
    assert "real.py" in names
