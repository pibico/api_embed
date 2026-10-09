"""api_embed — compute-only embedding + reranking service.

History: this service used to also store LEANN HNSW indexes per (site, library_id)
and expose /index, /search, /delete, /index/batch, /library, /libraries. Talk2Doc
Architecture 4+ moved vector storage into pgvector, leaving api_embed responsible
for two pure compute endpoints:

  - POST /api/v1/vectors  (multilingual-e5-large, 1024-dim, generate-only)
  - POST /api/v1/rerank   (bge-reranker-v2-m3, cross-encoder scoring)

Task #6 (2026-06-04) retires the dead LEANN surface: 6 endpoints + their
DuckDB-backed helpers + the on-disk indexes were untouched by any caller since
Talk2Doc's LEANN cutover (Task #2, commit 31fbb20). Keeping them around added
~430 LOC of attack surface, dragged leann + duckdb deps along, and held 43 MB
of stale HNSW files. Both endpoints retained here are lazy-loaded, so startup
stays light.
"""
from fastapi import FastAPI, HTTPException, Depends, Header, APIRouter
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import logging
import mimetypes
import os
import threading
from datetime import datetime


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['SENTENCE_TRANSFORMERS_BATCH_SIZE'] = '64'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_embed.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

import torch  # noqa: E402

GPU_AVAILABLE = torch.cuda.is_available()
GPU_DEVICE = 'cuda:0' if GPU_AVAILABLE else 'cpu'
logger.info(f"GPU Status: Available={GPU_AVAILABLE}, Device={GPU_DEVICE}")

# Proxy prefix — NGINX forwards /embed/* as-is to this app
PROXY_PREFIX = os.getenv('ROOT_PATH', '')

app = FastAPI(
    title="pibiCo Embedding API (compute-only)",
    description=(
        "Two-endpoint compute service for the Talk2Doc pgvector RAG: "
        "POST /api/v1/vectors (multilingual-e5-large, 1024d) and "
        "POST /api/v1/rerank (bge-reranker-v2-m3)."
    ),
    version="3.0.0",
    openapi_url=f"{PROXY_PREFIX}/api/v1/openapi.json",
    docs_url=None,  # Custom docs endpoint below
    redoc_url=f"{PROXY_PREFIX}/api/v1/redoc",
)

router = APIRouter(prefix=f"{PROXY_PREFIX}/api/v1")

API_TOKEN = "mi-embedding"
START_TIME = datetime.now()


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_TOKEN:
        logger.warning("Invalid API key attempted")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Custom Swagger UI (FastAPI's default doesn't know about PROXY_PREFIX).
# ─────────────────────────────────────────────────────────────────────────────
@app.get(f"{PROXY_PREFIX}/api/v1/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    openapi_url = f"{PROXY_PREFIX}/api/v1/openapi.json"
    static_prefix = f"{PROXY_PREFIX}/static"
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link type="text/css" rel="stylesheet" href="{static_prefix}/css/swagger-ui.css">
        <link rel="shortcut icon" href="{static_prefix}/icons/favicon.ico">
        <title>pibiCo Embedding API - Swagger UI</title>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="{static_prefix}/js/swagger-ui-bundle.js"></script>
        <script>
        const ui = SwaggerUIBundle({{
            url: '{openapi_url}',
            dom_id: '#swagger-ui',
            layout: 'BaseLayout',
            deepLinking: true,
            showExtensions: true,
            showCommonExtensions: true,
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIBundle.SwaggerUIStandalonePreset
            ],
        }})
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────────────────────────────────────
# /vectors — multilingual-e5-large, 1024-dim, normalize_embeddings=True.
# Caller prepends "query: " or "passage: " per the e5 convention (centralising
# the prefix server-side is a separate hardening item).
# ─────────────────────────────────────────────────────────────────────────────
_E5_MODEL = None
_E5_NAME = "intfloat/multilingual-e5-large"
# Un candado por modelo. El tokenizador rápido de HuggingFace no admite dos
# usos a la vez: el segundo revienta con «RuntimeError: Already borrowed», que
# sale como 500 y deja un lote entero sin vectores. Lo provocaban el recuento de
# truncado (en el bucle de eventos) contra el encode (en un hilo) y dos encode
# entre sí: el 08/10/2026 falló así el 26% de las llamadas a /vectors. Es una
# sola GPU, así que serializar no cuesta rendimiento real.
_E5_LOCK = threading.Lock()


def _get_e5_model():
    global _E5_MODEL
    if _E5_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s on %s", _E5_NAME, GPU_DEVICE)
        _E5_MODEL = SentenceTransformer(_E5_NAME, device=GPU_DEVICE)
    return _E5_MODEL


class VectorsRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = None  # advisory; this endpoint always serves e5-large
    # Task #6 (b) — opt-in: when set, the server prepends the e5 instruction
    # prefix internally so the caller stops being responsible for getting it
    # right. Legacy callers (no kind) still work — the server passes their
    # texts through untouched.
    kind: Optional[str] = None  # "query" | "passage" | None


def _e5_model_version() -> str:
    """Stable identifier for the loaded e5 weights. Prefers the HF commit SHA
    when sentence-transformers exposes it (lets the caller invalidate caches on
    a model swap); falls back to the model name."""
    try:
        m = _get_e5_model()
        # SentenceTransformer wraps a transformers AutoModel as the 0th module.
        auto = m._first_module().auto_model
        rev = getattr(auto.config, "_commit_hash", None) or getattr(auto.config, "commit_hash", None)
        if rev:
            return f"{_E5_NAME}@{str(rev)[:12]}"
    except Exception:
        pass
    return _E5_NAME


def _e5_truncation_flags(texts: List[str]) -> List[bool]:
    """One bool per text — True when the tokenizer's token count exceeds the
    model's max sequence length. Lets the caller surface what got silently
    dropped instead of discovering recall losses much later. Uses the live
    tokenizer; no add_special_tokens=True needed because we compare with the
    same +2 (CLS+SEP) overhead the model would add."""
    try:
        m = _get_e5_model()
        tok = m.tokenizer
        max_len = int(getattr(tok, "model_max_length", 512))
        out: List[bool] = []
        for t in texts:
            ids = tok.encode(t or "", add_special_tokens=False, truncation=False)
            out.append(len(ids) + 2 > max_len)
        return out
    except Exception:
        return [False] * len(texts)


@app.post(f"{PROXY_PREFIX}/api/v1/vectors", tags=["Embeddings"])
async def generate_vectors(req: VectorsRequest, _: bool = Depends(verify_api_key)):
    """Generate dense embeddings (no storage). Returns
    ``{vectors, model, model_version, dim, truncated}``.

    ``kind`` (Task #6 b): "query" or "passage" → server prepends the e5
    instruction prefix; unset → caller is responsible (legacy behaviour).
    Guarded against double-prefix: if a text already starts with the
    instruction, we skip the prepend so a mixed-deploy window can't corrupt
    embeddings."""
    if not req.texts:
        return {
            "vectors": [], "model": _E5_NAME,
            "model_version": _e5_model_version(),
            "dim": 1024, "truncated": [],
        }
    import anyio

    inputs = list(req.texts)
    if req.kind in ("query", "passage"):
        prefix = "query: " if req.kind == "query" else "passage: "
        inputs = [
            t if (t.startswith("query: ") or t.startswith("passage: ")) else prefix + t
            for t in inputs
        ]

    def _encode():
        # Recuento y encode bajo el mismo candado y en el mismo hilo: los dos
        # usan el tokenizador, y el recuento ya no bloquea el bucle de eventos.
        with _E5_LOCK:
            model = _get_e5_model()
            flags = _e5_truncation_flags(inputs)
            out = model.encode(
                inputs, normalize_embeddings=True, batch_size=32, show_progress_bar=False
            )
        return flags, out

    truncated, embs = await anyio.to_thread.run_sync(_encode)
    if any(truncated):
        n = sum(1 for x in truncated if x)
        logger.info(
            "vectors: %d/%d texts exceeded e5 max_seq_length and will be truncated",
            n, len(truncated),
        )
    vectors = [[float(x) for x in row] for row in embs]
    return {
        "vectors": vectors,
        "model": _E5_NAME,
        "model_version": _e5_model_version(),
        "dim": (len(vectors[0]) if vectors else 1024),
        "truncated": truncated,
    }


# ─────────────────────────────────────────────────────────────────────────────
# /rerank — bge-reranker-v2-m3 cross-encoder. One score per passage; higher is
# more relevant; raw logits, caller only sorts.
# ─────────────────────────────────────────────────────────────────────────────
_RERANK_MODEL = None
_RERANK_NAME = "BAAI/bge-reranker-v2-m3"
_RERANK_LOCK = threading.Lock()  # mismo motivo que _E5_LOCK


def _get_rerank_model():
    global _RERANK_MODEL
    if _RERANK_MODEL is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker %s on %s", _RERANK_NAME, GPU_DEVICE)
        _RERANK_MODEL = CrossEncoder(_RERANK_NAME, device=GPU_DEVICE, max_length=512)
    return _RERANK_MODEL


class RerankRequest(BaseModel):
    query: str
    passages: List[str]
    model: Optional[str] = None  # advisory; this endpoint always serves bge-reranker-v2-m3


@app.post(f"{PROXY_PREFIX}/api/v1/rerank", tags=["Rerank"])
async def rerank_passages(req: RerankRequest, _: bool = Depends(verify_api_key)):
    """Score each passage against the query with the cross-encoder.
    Returns {scores, model}; scores align 1:1 with req.passages."""
    if not req.passages:
        return {"scores": [], "model": _RERANK_NAME}
    import anyio
    pairs = [[req.query, (p or "")[:2000]] for p in req.passages]

    def _predict():
        with _RERANK_LOCK:
            return _get_rerank_model().predict(pairs, batch_size=16, show_progress_bar=False)

    scores = await anyio.to_thread.run_sync(_predict)
    return {"scores": [float(s) for s in scores], "model": _RERANK_NAME}


# ─────────────────────────────────────────────────────────────────────────────
# Health — no auth required. Reports the two real models served + GPU state.
# Placed after the model-name globals so it can reference _E5_NAME/_RERANK_NAME.
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    uptime = (datetime.now() - START_TIME).total_seconds()
    gpu_info = {"available": GPU_AVAILABLE, "device": GPU_DEVICE}
    if GPU_AVAILABLE:
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_info.update({
                "memory_allocated_mb": round(gpu_memory_allocated, 2),
                "memory_reserved_mb": round(gpu_memory_reserved, 2),
                "memory_total_mb": round(gpu_memory_total, 2),
                "memory_free_mb": round(gpu_memory_total - gpu_memory_reserved, 2),
                "device_name": torch.cuda.get_device_name(0),
            })
        except Exception as e:
            gpu_info["error"] = str(e)

    return {
        "status": "healthy",
        "service": "api_embed",
        "version": "3.0.0",
        "endpoints": {
            "embeddings": {
                "model": _E5_NAME,
                "dim": 1024,
                "path": f"{PROXY_PREFIX}/api/v1/vectors",
            },
            "rerank": {
                "model": _RERANK_NAME,
                "path": f"{PROXY_PREFIX}/api/v1/rerank",
            },
        },
        "gpu": gpu_info,
        "uptime_seconds": round(uptime, 2),
    }


# Mount the API router (carries /health under PROXY_PREFIX/api/v1)
app.include_router(router)

# Web splash page at PROXY_PREFIX/ (kept as-is; documents the live endpoints)
from app.api.v1.endpoints.web import router as web_router  # noqa: E402
app.include_router(web_router, prefix=PROXY_PREFIX, include_in_schema=False)


# Serve static assets for the swagger UI overrides
static_path = Path(__file__).parent.parent / "static"


@app.get(f"{PROXY_PREFIX}/static/{{file_path:path}}", include_in_schema=False)
async def serve_static(file_path: str):
    full_path = static_path / file_path
    if full_path.exists() and full_path.is_file():
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if file_path.endswith('.woff') or file_path.endswith('.woff2'):
            mime_type = 'font/woff2' if file_path.endswith('.woff2') else 'font/woff'
        return FileResponse(full_path, media_type=mime_type)
    return JSONResponse(content={"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6959)
