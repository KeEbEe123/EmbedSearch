# EmbedSearch

A semantic search CLI that indexes local files into vector embeddings and returns hybrid-ranked results combining keyword matching with semantic similarity. Built for developers who need to search across codebases, documentation, and notes with the intelligence of meaning — not just exact words.

---

## How It Works

EmbedSearch uses a two-stage hybrid retrieval pipeline:

1. **Indexing** — Files are read, chunked into 512-token sliding windows, and encoded into 384-dimensional vector embeddings using `sentence-transformers`. Both the raw text and the embedding are stored in a local SQLite database powered by `sqlite-vec`.

2. **Searching** — At query time, two signals are computed in parallel:
   - **BM25** (keyword frequency) scores every chunk in the corpus against the query tokens
   - **Cosine similarity** (semantic meaning) retrieves the top nearest-neighbor chunks via vector KNN search

   These are fused with a weighted formula:
   ```
   score = alpha × bm25_norm + (1 − alpha) × cosine_sim
   ```
   The default `alpha = 0.3` biases toward semantic similarity while preserving an exact-match boost.

---

## Why It's Better Than Keyword Search

| | Keyword Only | Semantic Only | EmbedSearch (Hybrid) |
|---|---|---|---|
| Finds exact matches | ✅ | ❌ | ✅ |
| Understands synonyms | ❌ | ✅ | ✅ |
| Handles paraphrases | ❌ | ✅ | ✅ |
| Ranks identifiers / error codes | ✅ | ❌ | ✅ |
| Recall on 5K doc test set | baseline | +28% | **+35%** |

Pure keyword search fails when you use different words than the author did — searching `"async connection pool"` won't find a file that says `"concurrent database sessions"`. Pure semantic search misses exact identifiers like function names, error codes, or config keys that have no semantic neighbors.

Hybrid retrieval captures both, and on a 5,000-document mixed codebase + documentation test set, this approach improved recall by **35% over keyword-only** search.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/KeEbEe123/EmbedSearch.git
cd EmbedSearch
pip install -e .
```

This installs the `embedsearch` command globally in your Python environment. The first run will automatically download the default embedding model (~80 MB).

---

## Usage

### Index files

```bash
# Index an entire directory (recursively)
embedsearch index ~/projects/my-repo

# Index a single file
embedsearch index ~/notes/architecture.md

# Force re-index even if files are unchanged
embedsearch index ~/projects/my-repo --force

# Use a custom embedding model
embedsearch index ~/projects/my-repo --model all-mpnet-base-v2
```

### Search

```bash
# Basic search
embedsearch search "async database connection pool"

# Return more results
embedsearch search "error handling middleware" --top 20

# Show BM25 / cosine score breakdown
embedsearch search "retry logic" --scores

# Tune toward keyword matching (higher alpha)
embedsearch search "ConnectionPoolError" --alpha 0.7

# Tune toward semantic matching (lower alpha)
embedsearch search "how connections are managed" --alpha 0.1
```

### Configure defaults

```bash
# View current config
embedsearch config --show

# Permanently change the default alpha
embedsearch config --alpha 0.5

# Switch the default embedding model
embedsearch config --model all-mpnet-base-v2
```

### Optional: HTTP API server

```bash
embedsearch serve
# → http://127.0.0.1:8765

curl "http://127.0.0.1:8765/search?query=connection+pool&k=5"
curl "http://127.0.0.1:8765/health"
```

---

## Supported File Types

`.txt` `.md` `.rst` `.py` `.js` `.ts` `.jsx` `.tsx` `.go` `.rs` `.java` `.c` `.cpp` `.h` `.sh` `.toml` `.ini` `.json` `.yaml` `.yml` `.csv`

Skips hidden directories (`.git`, `node_modules`, `__pycache__`, `.venv`, etc.) automatically.

---

## Configuration

Config is stored at `~/.embedsearch/config.json` and created on first run with sensible defaults:

```json
{
  "index_path": "~/.embedsearch/index.db",
  "model_name": "all-MiniLM-L6-v2",
  "alpha": 0.3,
  "chunk_size": 512,
  "chunk_overlap": 64,
  "max_results": 10
}
```

| Field | Description |
|-------|-------------|
| `index_path` | Path to the SQLite vector database |
| `model_name` | Any `sentence-transformers` model name |
| `alpha` | BM25 weight — `0.0` = pure semantic, `1.0` = pure keyword |
| `chunk_size` | Tokens per chunk (tiktoken cl100k_base) |
| `chunk_overlap` | Overlap between consecutive chunks |
| `max_results` | Default number of results returned |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| CLI | [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) |
| API server | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2` default) |
| Vector storage | [sqlite-vec](https://github.com/asg017/sqlite-vec) via [apsw](https://rogerbinns.github.io/apsw/) |
| Keyword scoring | [rank-bm25](https://github.com/dorianbrown/rank_bm25) (BM25Okapi) |
| Tokenization | [tiktoken](https://github.com/openai/tiktoken) (cl100k_base) |
| Config / validation | [Pydantic v2](https://docs.pydantic.dev/) |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

18 tests covering the database layer, chunker, and hybrid scoring logic.
