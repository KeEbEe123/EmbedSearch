from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from embedsearch import indexer, searcher
from embedsearch.config import Config
from embedsearch.db import ModelMismatchError
from embedsearch.models import IndexRequest, IndexResponse, SearchResponse

app = FastAPI(title="EmbedSearch", version="0.1.0")


@app.post("/index", response_model=IndexResponse)
def index_endpoint(req: IndexRequest) -> IndexResponse:
    cfg = Config.load()
    if req.model_name:
        cfg = cfg.model_copy(update={"model_name": req.model_name})
    t0 = time.perf_counter()
    try:
        stats = indexer.index_path(Path(req.path).resolve(), cfg, force=req.force)
    except ModelMismatchError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return IndexResponse(**stats, elapsed_seconds=time.perf_counter() - t0)


@app.get("/search", response_model=SearchResponse)
def search_endpoint(query: str, k: int = 10, alpha: float | None = None) -> SearchResponse:
    cfg = Config.load()
    if alpha is not None:
        cfg = cfg.model_copy(update={"alpha": alpha})
    t0 = time.perf_counter()
    try:
        results = searcher.hybrid_search(query, cfg, k)
    except ModelMismatchError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return SearchResponse(
        results=results,
        query=query,
        elapsed_seconds=time.perf_counter() - t0,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
