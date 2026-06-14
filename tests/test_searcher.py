"""Tests for hybrid search scoring logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embedsearch.searcher import _bm25_scores, _build_bm25, _cosine_scores


def test_bm25_scores_normalised():
    from rank_bm25 import BM25Okapi

    corpus = [
        "async database connection pool",
        "http request handler middleware",
        "async await coroutine event loop",
    ]
    ids = [1, 2, 3]
    bm25 = BM25Okapi([c.split() for c in corpus])

    scores = _bm25_scores(bm25, ids, "async database")
    values = list(scores.values())
    assert max(values) == pytest.approx(1.0)
    assert min(values) >= 0.0


def test_bm25_scores_empty_corpus():
    scores = _bm25_scores(None, [], "query")
    assert scores == {}


def test_hybrid_formula():
    """Verify score = alpha * bm25 + (1-alpha) * cosine."""
    alpha = 0.3
    bm25_score = 0.8
    cosine_score = 0.6
    expected = alpha * bm25_score + (1 - alpha) * cosine_score
    assert expected == pytest.approx(0.3 * 0.8 + 0.7 * 0.6)


def test_cosine_distance_to_similarity():
    """sqlite-vec distance [0,2] should convert to similarity [0,1]."""
    dist_identical = 0.0
    dist_orthogonal = 1.0
    dist_opposite = 2.0

    assert 1.0 - dist_identical / 2.0 == pytest.approx(1.0)
    assert 1.0 - dist_orthogonal / 2.0 == pytest.approx(0.5)
    assert 1.0 - dist_opposite / 2.0 == pytest.approx(0.0)
