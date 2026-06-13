from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Callable

import tiktoken
import yaml

from embedsearch.config import Config

_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".tox", ".venv", "venv", ".mypy_cache"}

_enc: tiktoken.Encoding | None = None


def _get_enc() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _read_plain(path: Path) -> str:
    return path.read_text(errors="replace")


def _read_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(errors="replace"))
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return path.read_text(errors="replace")


def _read_yaml(path: Path) -> str:
    try:
        data = yaml.safe_load(path.read_text(errors="replace"))
        return yaml.dump(data, allow_unicode=True)
    except Exception:
        return path.read_text(errors="replace")


def _read_csv(path: Path) -> str:
    try:
        text = path.read_text(errors="replace")
        reader = csv.reader(io.StringIO(text))
        return "\n".join(" | ".join(row) for row in reader)
    except Exception:
        return path.read_text(errors="replace")


READERS: dict[str, Callable[[Path], str]] = {
    ".txt": _read_plain,
    ".md": _read_plain,
    ".rst": _read_plain,
    ".py": _read_plain,
    ".js": _read_plain,
    ".ts": _read_plain,
    ".jsx": _read_plain,
    ".tsx": _read_plain,
    ".go": _read_plain,
    ".rs": _read_plain,
    ".java": _read_plain,
    ".c": _read_plain,
    ".cpp": _read_plain,
    ".h": _read_plain,
    ".sh": _read_plain,
    ".toml": _read_plain,
    ".ini": _read_plain,
    ".json": _read_json,
    ".yaml": _read_yaml,
    ".yml": _read_yaml,
    ".csv": _read_csv,
}


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    enc = _get_enc()
    tokens = enc.encode(text)
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return chunks


class ChunkRecord:
    __slots__ = ("path", "content", "chunk_idx", "chunk_text")

    def __init__(self, path: Path, content: str, chunk_idx: int, chunk_text: str):
        self.path = path
        self.content = content
        self.chunk_idx = chunk_idx
        self.chunk_text = chunk_text


def index_file(path: Path, cfg: Config) -> list[ChunkRecord]:
    ext = path.suffix.lower()
    reader = READERS.get(ext)
    if reader is None:
        return []
    try:
        content = reader(path)
    except Exception:
        return []
    content = content.strip()
    if not content:
        return []
    chunks = chunk_text(content, cfg.chunk_size, cfg.chunk_overlap)
    return [
        ChunkRecord(path, content, i, chunk)
        for i, chunk in enumerate(chunks)
    ]


def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in READERS else []
    files = []
    for p in target.rglob("*"):
        if p.is_file() and p.suffix.lower() in READERS:
            if not any(part in _SKIP_DIRS for part in p.parts):
                files.append(p)
    return files
