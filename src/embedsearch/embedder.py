from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_loaded_model_name: str | None = None


def get_model(model_name: str) -> SentenceTransformer:
    global _model, _loaded_model_name
    if _model is None or _loaded_model_name != model_name:
        _model = SentenceTransformer(model_name)
        _loaded_model_name = model_name
    return _model


def get_dim(model_name: str) -> int:
    return get_model(model_name).get_sentence_embedding_dimension()


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    model = get_model(model_name)
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(text: str, model_name: str) -> np.ndarray:
    model = get_model(model_name)
    vecs = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs[0]
