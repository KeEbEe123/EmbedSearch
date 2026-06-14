from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from embedsearch import db, embedder
from embedsearch.config import Config
from embedsearch.models import SearchResult


def _build_bm25(con) -> tuple[BM25Okapi | None, list[int]]:
    rows = db.get_all_chunks(con)
    if not rows:
        return None, []
    ids = [r["id"] for r in rows]
    tokenized = [r["chunk_text"].lower().split() for r in rows]
    return BM25Okapi(tokenized), ids


def _bm25_scores(bm25: BM25Okapi | None, ids: list[int], query: str) -> dict[int, float]:
    if not ids or bm25 is None:
        return {}
    tokens = query.lower().split()
    raw = bm25.get_scores(tokens)
    s_min, s_max = raw.min(), raw.max()
    if s_max == s_min:
        norm = np.zeros_like(raw)
    else:
        norm = (raw - s_min) / (s_max - s_min)
    return {doc_id: float(norm[i]) for i, doc_id in enumerate(ids)}


def _cosine_scores(con, query_vec: np.ndarray, k: int) -> dict[int, float]:
    rows = db.knn_search(con, query_vec, k)
    # sqlite-vec cosine distance [0,2] → similarity [0,1]
    return {doc_id: 1.0 - (dist / 2.0) for doc_id, dist in rows}


def hybrid_search(query: str, cfg: Config, k: int | None = None) -> list[SearchResult]:
    from embedsearch.db import get_connection

    if k is None:
        k = cfg.max_results

    dim = embedder.get_dim(cfg.model_name)
    con = get_connection(cfg.db_path(), dim, cfg.model_name)

    query_vec = embedder.embed_query(query, cfg.model_name)

    bm25, ids = _build_bm25(con)
    bm25_map = _bm25_scores(bm25, ids, query)
    # Over-retrieve from KNN to give BM25 candidates a cosine score too
    cosine_map = _cosine_scores(con, query_vec, k * 3)

    candidate_ids = set(bm25_map.keys()) | set(cosine_map.keys())
    if not candidate_ids:
        return []

    alpha = cfg.alpha
    scored = []
    for doc_id in candidate_ids:
        b = bm25_map.get(doc_id, 0.0)
        c = cosine_map.get(doc_id, 0.0)
        score = alpha * b + (1.0 - alpha) * c
        scored.append((doc_id, score, b, c))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    top_ids = [x[0] for x in top]
    chunks = db.get_chunks_by_ids(con, top_ids)
    chunk_map = {ch["id"]: ch for ch in chunks}

    results = []
    for doc_id, score, bm25_score, cosine_score in top:
        ch = chunk_map.get(doc_id)
        if ch is None:
            continue
        results.append(
            SearchResult(
                path=ch["path"],
                chunk_text=ch["chunk_text"],
                score=round(score, 6),
                bm25_score=round(bm25_score, 6),
                cosine_score=round(cosine_score, 6),
            )
        )
    return results
