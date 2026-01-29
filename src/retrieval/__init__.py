"""Hybrid Retrieval System for E2E Testing Agent.

This module implements industry-standard hybrid search combining:
- BM25 keyword search (PostgreSQL Full-Text Search)
- Vector semantic search (pgvector)
- Reciprocal Rank Fusion (RRF) for merging results
- Cross-encoder reranking for final ranking

Why Hybrid Search?
------------------
Single retrieval methods have limitations:
- BM25 alone: Misses semantic similarity (e.g., "button not found" vs "element missing")
- Vector alone: Misses exact matches (e.g., specific error codes)
- Hybrid: Best of both worlds

Industry Adoption:
------------------
- Google Vertex AI: Hybrid search is the default
- Microsoft GraphRAG: Combines vector + semantic ranking + graph
- Weaviate/Qdrant: Native hybrid search is a key differentiator

Performance Benefits:
--------------------
- 20-30% improvement in retrieval quality vs single-method
- Better handling of both conceptual and exact matches
- More robust to query phrasing variations
"""

from .codebase_retriever import (
    CodebaseRetriever,
    CodeSearchResult,
    SearchSource,
    get_codebase_retriever,
    init_codebase_retriever,
    reset_codebase_retriever,
)
from .cross_encoder import CrossEncoderReranker
from .hybrid_retriever import (
    HybridRetriever,
    RetrievalResult,
    RetrievalSource,
    get_hybrid_retriever,
    init_hybrid_retriever,
    reset_hybrid_retriever,
)

__all__ = [
    # Failure pattern retriever
    "HybridRetriever",
    "RetrievalResult",
    "RetrievalSource",
    "CrossEncoderReranker",
    "get_hybrid_retriever",
    "init_hybrid_retriever",
    "reset_hybrid_retriever",
    # Codebase retriever
    "CodebaseRetriever",
    "CodeSearchResult",
    "SearchSource",
    "get_codebase_retriever",
    "init_codebase_retriever",
    "reset_codebase_retriever",
]
