# api_embed — compute-only embedding & reranking service

Two-endpoint GPU compute service for the Talk2Doc pgvector RAG pipeline. It
generates dense vectors and scores query/passage pairs. It stores nothing.

- `POST /api/v1/vectors` — `intfloat/multilingual-e5-large`, 1024-dim
- `POST /api/v1/rerank` — `BAAI/bge-reranker-v2-m3` cross-encoder

Both models are lazy-loaded on first request, so startup stays light.

## History

Until v3.0.0 this service also stored LEANN HNSW indexes per `(site, library_id)`
and exposed `/index`, `/search`, `DELETE /index`, `/index/batch`,
`DELETE /library` and `/libraries`, backed by DuckDB.

Talk2Doc Architecture 4+ moved vector storage into pgvector, and after the LEANN
cutover (Task #2, commit `31fbb20`) no caller touched that surface again. Task #6
(2026-06-04) retired it: ~430 LOC of dead attack surface, the `leann` and `duckdb`
dependencies, and 43 MB of stale HNSW files are gone. **If you are looking for
the search/index API, it no longer exists here — query pgvector in Talk2Doc.**

## Endpoints

NGINX forwards `/embed/*` to this app unchanged, and `ROOT_PATH=/embed` makes the
app mount itself under that prefix. Paths below are as seen from outside.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/embed/api/v1/vectors` | `X-API-Key` | Generate embeddings |
| POST | `/embed/api/v1/rerank` | `X-API-Key` | Score passages against a query |
| GET | `/embed/api/v1/health` | none | Models served, GPU state, uptime |
| GET | `/embed/api/v1/docs` | none | Swagger UI (local assets, no CDN) |
| GET | `/embed/api/v1/redoc` | none | ReDoc |
| GET | `/embed/api/v1/openapi.json` | none | OpenAPI schema |
| GET | `/embed/` | none | Splash page describing the live endpoints |

### POST /api/v1/vectors

```json
{ "texts": ["¿qué dice el contrato?"], "kind": "query" }
```

- `texts` — list of strings. Empty list returns an empty result, not an error.
- `kind` — `"query"` | `"passage"` | omitted. When set, the server prepends the
  e5 instruction prefix (`query: ` / `passage: `) itself. It is guarded against
  double-prefixing: a text that already carries either prefix is passed through
  untouched, so a mixed-deploy window cannot corrupt embeddings. When omitted,
  the caller is responsible for the prefix (legacy behaviour).
- `model` — advisory only; this endpoint always serves e5-large.

Response:

```json
{
  "vectors": [[0.01, ...]],
  "model": "intfloat/multilingual-e5-large",
  "model_version": "intfloat/multilingual-e5-large@<hf-commit-sha>",
  "dim": 1024,
  "truncated": [false]
}
```

- Vectors are L2-normalised (`normalize_embeddings=True`), encoded in batches of 32.
- `model_version` prefers the Hugging Face commit SHA of the loaded weights so
  callers can invalidate caches when the model is swapped; it falls back to the
  bare model name when sentence-transformers does not expose it.
- `truncated` carries one bool per input text, `true` when the text exceeds the
  model's 512-token limit and was silently cut. Surface it — otherwise
  truncation shows up much later as unexplained recall loss.

### POST /api/v1/rerank

```json
{ "query": "plazo de entrega", "passages": ["...", "..."] }
```

Returns `{"scores": [...], "model": "BAAI/bge-reranker-v2-m3"}`, aligned 1:1 with
`passages`. Scores are **raw cross-encoder logits**, not probabilities — higher is
more relevant, and the caller only sorts by them. Passages are clipped to 2000
characters before scoring and processed in batches of 16. An empty `passages`
list returns empty scores.

### Auth

`/vectors` and `/rerank` require an `X-API-Key` header; a mismatch returns 401.
`/health` and the docs endpoints are open.

> Hardening item: the expected key is currently a hardcoded constant in
> `app/main.py` rather than an environment variable. Move it to the environment
> before this service is reachable from anywhere new.

## Deployment (host `1BC1CD8`)

This service runs **directly from this tree** — there is no separate release
checkout. Editing files here changes production.

| | |
|---|---|
| Code | `/home/erpnext/.services/api_embed` |
| Virtualenv | `/home/erpnext/api_embed_env` |
| Process manager | supervisor, `/etc/supervisor/conf.d/api_embed.conf` |
| Bind | `127.0.0.1:6959`, uvicorn, `--workers 1`, `--timeout-keep-alive 600` |
| Public route | `api.espib.co/embed/` → `127.0.0.1:6959` (`/etc/nginx/conf.d/api.espib.co.conf`) |
| GPU | CUDA `cuda:0`, with automatic CPU fallback when unavailable |

Environment set by supervisor: `ROOT_PATH=/embed`,
`HF_HOME=/home/erpnext/.cache/huggingface`, `TOKENIZERS_PARALLELISM=false`,
plus the venv `PATH` and `PYTHONPATH`.

A single worker is deliberate: both models live in one process's GPU memory, and
more workers would multiply the VRAM footprint for no throughput gain. Encoding
runs in a worker thread (`anyio.to_thread`) so the event loop stays responsive.

### Deploy = edit in place, then restart

```bash
sudo supervisorctl restart api_embed
sudo supervisorctl status api_embed
curl -s http://127.0.0.1:6959/embed/api/v1/health | jq
```

Logs: `logs/api_embed.log` in this directory, plus supervisor's own output.

### Local development

```bash
python3 -m venv /home/erpnext/api_embed_env
/home/erpnext/api_embed_env/bin/pip install -r requirements.txt
ROOT_PATH=/embed /home/erpnext/api_embed_env/bin/uvicorn app.main:app \
  --host 127.0.0.1 --port 6959 --reload
```

First request to each endpoint downloads and loads its model — expect a slow
first call, fast ones after.

## Consumers

`api_talk2doc` is the only caller. It reaches both endpoints from
`app/clients/embed_client.py` (vectors), `app/services/rerank.py` (rerank), with
the URLs configured in `app/config.py`.

Changing a model, the vector dimension, or the response shape breaks Talk2Doc
retrieval. The 1024-dim output is fixed by the pgvector column on that side, so
an embedding-model swap is a coordinated migration, never a drop-in.

## Stack

FastAPI · uvicorn · sentence-transformers · PyTorch (CUDA)
