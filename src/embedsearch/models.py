from pydantic import BaseModel


class IndexRequest(BaseModel):
    path: str
    force: bool = False
    model_name: str | None = None


class IndexResponse(BaseModel):
    files_indexed: int
    chunks_created: int
    elapsed_seconds: float


class SearchResult(BaseModel):
    path: str
    chunk_text: str
    score: float
    bm25_score: float
    cosine_score: float


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    alpha: float | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    elapsed_seconds: float
