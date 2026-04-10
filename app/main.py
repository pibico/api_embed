from fastapi import FastAPI, HTTPException, Depends, Header, APIRouter
from pydantic import BaseModel
from leann import LeannBuilder, LeannSearcher
from pathlib import Path
import logging
import json
import os
import time
from functools import lru_cache
from hashlib import md5
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from datetime import datetime
import duckdb
import shutil

# GPU OPTIMIZATION: Enable GPU with memory controls
# Using GPU with batch size limits to prevent OOM
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA for performance
# Batch size control for sentence-transformers (optimized for RTX 3060)
os.environ['SENTENCE_TRANSFORMERS_BATCH_SIZE'] = '64'  # Optimized for 12GB VRAM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/pi/.services/api_embed/logs/api_embed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize GPU after logger is configured
import torch
GPU_AVAILABLE = torch.cuda.is_available()
GPU_DEVICE = 'cuda:0' if GPU_AVAILABLE else 'cpu'
logger.info(f"GPU Status: Available={GPU_AVAILABLE}, Device={GPU_DEVICE}")

# Performance metrics storage
performance_metrics = {
    'search_count': 0,
    'index_count': 0,
    'total_search_time': 0.0,
    'total_index_time': 0.0,
    'errors': defaultdict(int),
    'start_time': datetime.now()
}

# Proxy prefix — NGINX forwards /embed/* as-is to this app
PROXY_PREFIX = os.getenv('ROOT_PATH', '')

app = FastAPI(
    title="pibiRAG Unified Embedding API (Multi-Tenant)",
    description="Unified API for document embedding and semantic search using LEANN with multi-tenant support",
    version="2.1.0",
    openapi_url=f"{PROXY_PREFIX}/api/v1/openapi.json",
    docs_url=None,  # Custom docs endpoint below
    redoc_url=f"{PROXY_PREFIX}/api/v1/redoc",
)

# Create router with prefix (NGINX passes /embed/api/v1/* as-is)
router = APIRouter(prefix=f"{PROXY_PREFIX}/api/v1")

# Base directory for LEANN indexes
INDEX_BASE_DIR = Path("/home/pi/.services/api_embed/data/indexes")
INDEX_BASE_DIR.mkdir(parents=True, exist_ok=True)

# API Token
API_TOKEN = "mi-embedding"

# Configuration
MAX_CONTENT_CHARS = 8192  # Increased from 2048 for better context
SEARCH_CACHE_SIZE = 1000  # Number of cached search results
SEARCH_TIMEOUT = 30  # seconds

# Pydantic Models
class IndexRequest(BaseModel):
    site: str  # e.g., "dev.pibico.es"
    library_id: str  # e.g., "APPS_LIBRARY"
    doc_name: str
    content: str
    metadata: dict = {}

class SearchRequest(BaseModel):
    site: str
    library_id: str
    query: str
    k: int = 10
    full_text: bool = False  # Return full document text instead of 200-char snippet

class DeleteRequest(BaseModel):
    site: str
    library_id: str
    doc_name: str

class ChunkItem(BaseModel):
    doc_name: str       # "chunk_0", "chunk_1", ...
    content: str
    metadata: dict = {}

class BatchIndexRequest(BaseModel):
    site: str
    library_id: str
    chunks: List[ChunkItem]
    replace: bool = True  # Delete existing library first

class LibraryDeleteRequest(BaseModel):
    site: str
    library_id: str


# Authentication
def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_TOKEN:
        logger.warning(f"Invalid API key attempted")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def get_index_path(site: str, library_id: str) -> Path:
    """Get the index path for a site + library"""
    # Sanitize site name (replace : with _)
    safe_site = site.replace(":", "_")
    return INDEX_BASE_DIR / safe_site / library_id / "index"

def get_duckdb_path(index_path: Path) -> Path:
    """Get DuckDB file path for a library"""
    return index_path.parent / "documents.duckdb"

def init_duckdb(db_path: Path):
    """Initialize DuckDB schema with optimized settings"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))

    # Create documents table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_name TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSON,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for fast queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_name ON documents(doc_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_indexed_at ON documents(indexed_at)")

    conn.close()
    logger.info(f"DuckDB initialized at {db_path}")

def load_existing_documents(index_path: Path) -> List[Tuple[str, Dict]]:
    """
    Load existing documents from DuckDB
    Returns list of (text, metadata) tuples
    """
    db_path = get_duckdb_path(index_path)
    documents = []

    if db_path.exists():
        logger.info(f"Loading existing documents from DuckDB: {db_path}")
        try:
            conn = duckdb.connect(str(db_path), read_only=True)
            results = conn.execute("""
                SELECT content, metadata
                FROM documents
                ORDER BY indexed_at
            """).fetchall()
            conn.close()

            for row in results:
                content = row[0]
                metadata = json.loads(row[1]) if row[1] else {}
                documents.append((content, metadata))

            logger.info(f"Loaded {len(documents)} existing documents from DuckDB")
        except Exception as e:
            logger.error(f"Error loading from DuckDB: {e}", exc_info=True)
            return []

    return documents

def rebuild_index_with_documents(index_path: Path, documents: List[Tuple[str, Dict]]):
    """
    Rebuild LEANN index from scratch with all documents
    Saves documents to DuckDB and builds LEANN vector index
    Enhanced with GPU memory management and performance monitoring
    """
    if not documents:
        logger.warning("No documents to index")
        return

    start_time = time.time()
    logger.info(f"Rebuilding index with {len(documents)} documents on {GPU_DEVICE}")

    # Clear GPU cache before starting if GPU is available
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
        logger.info(f"GPU memory cleared before indexing")

    try:
        # Initialize DuckDB
        db_path = get_duckdb_path(index_path)
        init_duckdb(db_path)

        # Save all documents to DuckDB
        conn = duckdb.connect(str(db_path))
        for text, metadata in documents:
            doc_name = metadata.get('doc_name', 'unknown')
            conn.execute("""
                INSERT OR REPLACE INTO documents (doc_name, content, metadata, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (doc_name, text, json.dumps(metadata)))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(documents)} documents to DuckDB")

        # Create new builder (will use GPU if available)
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="BAAI/bge-base-en-v1.5",
            dimensions=768,
            embedding_mode="sentence-transformers"
        )

        # Add all documents to LEANN
        for text, metadata in documents:
            builder.add_text(text=text, metadata=metadata)

        # Build LEANN index
        index_path.parent.mkdir(parents=True, exist_ok=True)
        builder.build_index(index_path=str(index_path))

        # Clear GPU cache after indexing
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        logger.info(f"Index rebuilt successfully at {index_path} in {elapsed:.2f}s")

        # Log GPU memory usage if available
        if GPU_AVAILABLE:
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            logger.info(f"GPU memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")

    except RuntimeError as e:
        elapsed = time.time() - start_time
        if "out of memory" in str(e).lower():
            logger.error(f"GPU OOM during index rebuild after {elapsed:.2f}s", exc_info=True)
            # Clear cache and retry on CPU
            if GPU_AVAILABLE:
                torch.cuda.empty_cache()
            logger.warning("Falling back to CPU mode for this operation")
        performance_metrics['errors']['index_rebuild'] += 1
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Index rebuild failed after {elapsed:.2f}s: {e}", exc_info=True)
        performance_metrics['errors']['index_rebuild'] += 1
        raise

# Custom Swagger UI endpoint with correct OpenAPI URL
@app.get(f"{PROXY_PREFIX}/api/v1/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI that works with proxy prefix."""
    from fastapi.responses import HTMLResponse
    openapi_url = f"{PROXY_PREFIX}/api/v1/openapi.json"
    static_prefix = f"{PROXY_PREFIX}/static"
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link type="text/css" rel="stylesheet" href="{static_prefix}/css/swagger-ui.css">
        <link rel="shortcut icon" href="{static_prefix}/icons/favicon.ico">
        <title>pibiRAG Embedding API - Swagger UI</title>
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

# Health check (no auth required)
@router.get("/health")
async def health():
    """Enhanced health check endpoint with performance metrics and GPU status"""
    uptime = (datetime.now() - performance_metrics['start_time']).total_seconds()

    avg_search_time = (
        performance_metrics['total_search_time'] / performance_metrics['search_count']
        if performance_metrics['search_count'] > 0 else 0
    )

    avg_index_time = (
        performance_metrics['total_index_time'] / performance_metrics['index_count']
        if performance_metrics['index_count'] > 0 else 0
    )

    # GPU status
    gpu_info = {
        "available": GPU_AVAILABLE,
        "device": GPU_DEVICE
    }

    if GPU_AVAILABLE:
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            gpu_info.update({
                "memory_allocated_mb": round(gpu_memory_allocated, 2),
                "memory_reserved_mb": round(gpu_memory_reserved, 2),
                "memory_total_mb": round(gpu_memory_total, 2),
                "memory_free_mb": round(gpu_memory_total - gpu_memory_reserved, 2),
                "device_name": torch.cuda.get_device_name(0)
            })
        except Exception as e:
            gpu_info["error"] = str(e)

    return {
        "status": "healthy",
        "service": "api_embed",
        "model": "BAAI/bge-base-en-v1.5 (via LEANN)",
        "dimension": 768,
        "backend": f"LEANN HNSW ({GPU_DEVICE})",
        "multi_tenant": True,
        "port": 6959,
        "version": "2.1.0",
        "gpu": gpu_info,
        "performance": {
            "uptime_seconds": round(uptime, 2),
            "total_searches": performance_metrics['search_count'],
            "total_indexing": performance_metrics['index_count'],
            "avg_search_time_ms": round(avg_search_time * 1000, 2),
            "avg_index_time_ms": round(avg_index_time * 1000, 2),
            "error_count": sum(performance_metrics['errors'].values()),
            "errors_by_type": dict(performance_metrics['errors'])
        },
        "config": {
            "max_content_chars": MAX_CONTENT_CHARS,
            "search_cache_size": SEARCH_CACHE_SIZE,
            "search_timeout": SEARCH_TIMEOUT,
            "batch_size": 64,
            "storage": "DuckDB",
            "gpu_enabled": GPU_AVAILABLE
        }
    }

@router.post("/index")
async def index_document(req: IndexRequest, _: bool = Depends(verify_api_key)):
    """
    Index a document: Store text in LEANN (LEANN generates embeddings internally)

    Multi-tenant: site + library_id creates separate indexes
    Note: LEANN compact HNSW doesn't support incremental updates, so we rebuild the entire index
    Enhanced with performance monitoring and increased content limit
    """
    start_time = time.time()
    try:
        logger.info(f"Indexing document {req.doc_name} in {req.site}:{req.library_id}")

        # Validate content
        if not req.content or not req.content.strip():
            raise HTTPException(status_code=400, detail="Empty content")

        # Truncate content if too large (increased from 2048 to 8192)
        content = req.content
        if len(content) > MAX_CONTENT_CHARS:
            logger.warning(f"Content too large ({len(content)} chars), truncating to {MAX_CONTENT_CHARS}")
            content = content[:MAX_CONTENT_CHARS]

        # Get index path
        index_path = get_index_path(req.site, req.library_id)

        # Load existing documents
        existing_docs = load_existing_documents(index_path)

        # Check if document already exists (by doc_name in metadata)
        existing_doc_names = {doc[1].get('doc_name') for doc in existing_docs if doc[1].get('doc_name')}

        # Prepare new document
        new_text = f"[{req.doc_name}] {content}"
        new_metadata = {"doc_name": req.doc_name, **req.metadata}

        # If document exists, remove old version
        if req.doc_name in existing_doc_names:
            logger.info(f"Document {req.doc_name} already exists, replacing")
            existing_docs = [(text, meta) for text, meta in existing_docs
                           if meta.get('doc_name') != req.doc_name]

        # Add new document
        existing_docs.append((new_text, new_metadata))

        # Invalidate search cache for this library
        invalidate_cache_for_library(req.site, req.library_id)

        # Rebuild index with all documents
        rebuild_index_with_documents(index_path, existing_docs)

        # Update metrics
        elapsed = time.time() - start_time
        performance_metrics['index_count'] += 1
        performance_metrics['total_index_time'] += elapsed

        logger.info(f"Successfully indexed {req.doc_name} in {req.site}:{req.library_id} ({elapsed:.2f}s)")
        return {
            "status": "success",
            "doc_name": req.doc_name,
            "site": req.site,
            "library_id": req.library_id,
            "content_length": len(content),
            "total_documents": len(existing_docs),
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    except Exception as e:
        elapsed = time.time() - start_time
        performance_metrics['errors']['index'] += 1
        logger.error(f"Failed to index document {req.doc_name} after {elapsed:.2f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

# Cache for search results
search_cache = {}

def get_search_cache_key(site: str, library_id: str, query: str, k: int) -> str:
    """Generate cache key for search query"""
    cache_str = f"{site}:{library_id}:{query}:{k}"
    return md5(cache_str.encode()).hexdigest()

def invalidate_cache_for_library(site: str, library_id: str):
    """Invalidate all cached search results for a specific library"""
    keys_to_remove = []
    library_prefix = f"{site}:{library_id}:"

    for key, value in search_cache.items():
        # Check if this cache entry belongs to the library
        if value.get('site') == site and value.get('library_id') == library_id:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del search_cache[key]

    if keys_to_remove:
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for {site}:{library_id}")

@router.post("/search")
async def search(req: SearchRequest, _: bool = Depends(verify_api_key)):
    """
    Semantic search: Search LEANN index with text query

    Multi-tenant: searches within site + library_id scope
    Enhanced with caching, timeout protection, and performance monitoring
    """
    start_time = time.time()
    try:
        logger.info(f"Searching in {req.site}:{req.library_id} with query: '{req.query[:50]}...'")

        # Check cache first
        cache_key = get_search_cache_key(req.site, req.library_id, req.query, req.k)
        if cache_key in search_cache:
            cached_result = search_cache[cache_key]
            elapsed = time.time() - start_time
            logger.info(f"Cache hit for {req.site}:{req.library_id} ({elapsed*1000:.2f}ms)")
            cached_result['cached'] = True
            cached_result['processing_time_ms'] = round(elapsed * 1000, 2)
            return cached_result

        # Get index path
        index_path = get_index_path(req.site, req.library_id)
        index_file = Path(str(index_path) + ".index")

        if not index_file.exists():
            logger.warning(f"No index found for {req.site}:{req.library_id}")
            return {
                "status": "success",
                "results": [],
                "site": req.site,
                "library_id": req.library_id,
                "query": req.query,
                "cached": False
            }

        # Search LEANN (it generates query embeddings internally)
        searcher = LeannSearcher(index_path=str(index_path))
        results = searcher.search(query=req.query, top_k=req.k)

        # Format results
        formatted = []
        for result in results:
            # Extract doc_name from metadata
            doc_name = result.metadata.get("doc_name", "unknown")
            text_value = None
            if hasattr(result, 'text'):
                text_value = result.text if req.full_text else result.text[:200]
            formatted.append({
                "doc_name": doc_name,
                "similarity": float(result.score),
                "text_snippet": text_value
            })

        # Update metrics
        elapsed = time.time() - start_time
        performance_metrics['search_count'] += 1
        performance_metrics['total_search_time'] += elapsed

        # Prepare response
        response = {
            "status": "success",
            "results": formatted,
            "site": req.site,
            "library_id": req.library_id,
            "query": req.query,
            "total": len(formatted),
            "cached": False,
            "processing_time_ms": round(elapsed * 1000, 2)
        }

        # Cache the result (simple LRU: clear if cache is too large)
        if len(search_cache) >= SEARCH_CACHE_SIZE:
            # Remove oldest entries (first half)
            keys_to_remove = list(search_cache.keys())[:SEARCH_CACHE_SIZE // 2]
            for key in keys_to_remove:
                del search_cache[key]
            logger.info(f"Cache cleared: removed {len(keys_to_remove)} entries")

        search_cache[cache_key] = response.copy()

        logger.info(f"Found {len(formatted)} results in {req.site}:{req.library_id} ({elapsed:.2f}s)")
        return response

    except Exception as e:
        elapsed = time.time() - start_time
        performance_metrics['errors']['search'] += 1
        logger.error(f"Search failed in {req.site}:{req.library_id} after {elapsed:.2f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")



@router.delete("/index")
async def delete_document(req: DeleteRequest, _: bool = Depends(verify_api_key)):
    """
    Delete a document from the index
    
    Since LEANN compact HNSW doesn't support in-place deletion,
    we clear the entire library index directory. The next index
    operation will rebuild the index from current Frappe state.
    """
    try:
        logger.info(f"Deleting document {req.doc_name} from {req.site}:{req.library_id}")
        
        index_path = get_index_path(req.site, req.library_id)
        index_dir = index_path.parent
        
        if not index_dir.exists():
            logger.warning(f"No index found for {req.site}:{req.library_id}")
            return {
                "status": "success",
                "message": "Index does not exist",
                "site": req.site,
                "library_id": req.library_id,
                "doc_name": req.doc_name,
                "deleted": False
            }

        # Invalidate search cache for this library
        invalidate_cache_for_library(req.site, req.library_id)

        # Clear the entire library index directory
        shutil.rmtree(index_dir)
        logger.info(f"Cleared index directory: {index_dir}")
        
        return {
            "status": "success",
            "message": f"Library index cleared",
            "site": req.site,
            "library_id": req.library_id,
            "doc_name": req.doc_name,
            "deleted": True
        }
        
    except Exception as e:
        logger.error(f"Failed to delete document {req.doc_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.post("/index/batch")
async def batch_index_documents(req: BatchIndexRequest, _: bool = Depends(verify_api_key)):
    """
    Batch index multiple chunks into a library at once.

    If replace=True (default), wipes the existing library first.
    All chunks are indexed in a single LEANN rebuild for efficiency.
    """
    start_time = time.time()
    try:
        logger.info(f"Batch indexing {len(req.chunks)} chunks into {req.site}:{req.library_id} (replace={req.replace})")

        if not req.chunks:
            raise HTTPException(status_code=400, detail="No chunks provided")

        index_path = get_index_path(req.site, req.library_id)

        # If replace mode, wipe existing library
        if req.replace:
            index_dir = index_path.parent
            if index_dir.exists():
                shutil.rmtree(index_dir)
                logger.info(f"Cleared existing library directory: {index_dir}")

        # Invalidate search cache for this library
        invalidate_cache_for_library(req.site, req.library_id)

        # Prepare all chunks as (text, metadata) tuples
        documents = []
        for chunk in req.chunks:
            if not chunk.content or not chunk.content.strip():
                continue
            content = chunk.content[:MAX_CONTENT_CHARS] if len(chunk.content) > MAX_CONTENT_CHARS else chunk.content
            text = f"[{chunk.doc_name}] {content}"
            metadata = {"doc_name": chunk.doc_name, **chunk.metadata}
            documents.append((text, metadata))

        if not documents:
            raise HTTPException(status_code=400, detail="All chunks were empty")

        # Rebuild index with all documents at once
        rebuild_index_with_documents(index_path, documents)

        elapsed = time.time() - start_time
        performance_metrics['index_count'] += 1
        performance_metrics['total_index_time'] += elapsed

        logger.info(f"Batch indexed {len(documents)} chunks into {req.site}:{req.library_id} ({elapsed:.2f}s)")
        return {
            "status": "success",
            "site": req.site,
            "library_id": req.library_id,
            "chunks_indexed": len(documents),
            "total_documents": len(documents),
            "processing_time_ms": round(elapsed * 1000, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        performance_metrics['errors']['batch_index'] += 1
        logger.error(f"Batch indexing failed for {req.site}:{req.library_id} after {elapsed:.2f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch indexing failed: {str(e)}")

@router.delete("/library")
async def delete_library(req: LibraryDeleteRequest, _: bool = Depends(verify_api_key)):
    """
    Delete an entire library (all documents and index).

    Removes the library directory and invalidates search cache.
    """
    try:
        logger.info(f"Deleting library {req.site}:{req.library_id}")

        index_path = get_index_path(req.site, req.library_id)
        index_dir = index_path.parent

        if not index_dir.exists():
            return {
                "status": "success",
                "message": "Library does not exist",
                "site": req.site,
                "library_id": req.library_id,
                "deleted": False
            }

        # Invalidate search cache
        invalidate_cache_for_library(req.site, req.library_id)

        # Remove entire library directory
        shutil.rmtree(index_dir)
        logger.info(f"Deleted library directory: {index_dir}")

        return {
            "status": "success",
            "message": "Library deleted",
            "site": req.site,
            "library_id": req.library_id,
            "deleted": True
        }

    except Exception as e:
        logger.error(f"Failed to delete library {req.site}:{req.library_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Library delete failed: {str(e)}")

@router.get("/libraries")
async def list_libraries(_: bool = Depends(verify_api_key)):
    """List all indexed libraries across all sites"""
    try:
        libraries = []

        # Iterate through site directories
        for site_dir in INDEX_BASE_DIR.iterdir():
            if site_dir.is_dir():
                site_name = site_dir.name

                # Iterate through library directories
                for lib_dir in site_dir.iterdir():
                    if lib_dir.is_dir():
                        # Check for DuckDB file
                        db_file = lib_dir / "documents.duckdb"
                        index_file = lib_dir / "index.index"

                        if db_file.exists():
                            # Count documents from DuckDB
                            doc_count = 0
                            db_size = 0
                            last_modified = 0

                            try:
                                conn = duckdb.connect(str(db_file), read_only=True)
                                result = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
                                doc_count = result[0] if result else 0

                                # Get last updated timestamp
                                result = conn.execute("SELECT MAX(updated_at) FROM documents").fetchone()
                                if result and result[0]:
                                    last_modified = result[0].timestamp() if hasattr(result[0], 'timestamp') else 0
                                conn.close()

                                db_size = db_file.stat().st_size
                            except Exception as e:
                                logger.error(f"Error reading DuckDB {db_file}: {e}")

                            libraries.append({
                                "site": site_name,
                                "library_id": lib_dir.name,
                                "full_id": f"{site_name}:{lib_dir.name}",
                                "document_count": doc_count,
                                "db_size": db_size,
                                "index_size": index_file.stat().st_size if index_file.exists() else 0,
                                "last_modified": last_modified or db_file.stat().st_mtime
                            })

        logger.info(f"Found {len(libraries)} libraries with {sum(lib['document_count'] for lib in libraries)} total documents")
        return {
            "status": "success",
            "libraries": libraries,
            "total": len(libraries)
        }

    except Exception as e:
        logger.error(f"List libraries failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Include API router in app
app.include_router(router)

# Include web interface at proxy prefix
from app.api.v1.endpoints.web import router as web_router
app.include_router(web_router, prefix=PROXY_PREFIX, include_in_schema=False)

# Serve static files
from fastapi.responses import FileResponse
from pathlib import Path
import mimetypes

static_path = Path(__file__).parent.parent / "static"

@app.get(f"{PROXY_PREFIX}/static/{{file_path:path}}", include_in_schema=False)
async def serve_static(file_path: str):
    """Serve static files."""
    full_path = static_path / file_path
    if full_path.exists() and full_path.is_file():
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if file_path.endswith('.woff') or file_path.endswith('.woff2'):
            mime_type = 'font/woff2' if file_path.endswith('.woff2') else 'font/woff'
        return FileResponse(full_path, media_type=mime_type)
    from fastapi.responses import JSONResponse
    return JSONResponse(content={"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6959)
