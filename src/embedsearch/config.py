from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_CONFIG_DIR = Path.home() / ".embedsearch"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_DEFAULTS = {
    "index_path": str(_CONFIG_DIR / "index.db"),
    "model_name": "all-MiniLM-L6-v2",
    "alpha": 0.3,
    "chunk_size": 512,
    "chunk_overlap": 64,
    "max_results": 10,
}


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index_path: str = _DEFAULTS["index_path"]
    model_name: str = _DEFAULTS["model_name"]
    alpha: float = _DEFAULTS["alpha"]
    chunk_size: int = _DEFAULTS["chunk_size"]
    chunk_overlap: int = _DEFAULTS["chunk_overlap"]
    max_results: int = _DEFAULTS["max_results"]

    def db_path(self) -> Path:
        return Path(self.index_path).expanduser().resolve()

    @classmethod
    def load(cls) -> "Config":
        if _CONFIG_FILE.exists():
            data = json.loads(_CONFIG_FILE.read_text())
            merged = {**_DEFAULTS, **data}
        else:
            merged = _DEFAULTS.copy()
        return cls(**merged)

    @staticmethod
    def save(cfg: "Config") -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(cfg.model_dump_json(indent=2))
