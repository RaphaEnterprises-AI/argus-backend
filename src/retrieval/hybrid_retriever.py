"""Hybrid Retriever combining BM25 + Vector Search + Reranking.

This module implements the complete hybrid retrieval pipeline:

1. BM25 Keyword Search (PostgreSQL Full-Text Search)
   - Exact term matching
   - Handles specific error codes, selectors
   - Fast and deterministic

2. Vector Semantic Search (pgvector)
   - Conceptual similarity
   - Handles paraphrasing and synonyms
   - Embeddings from OpenAI

3. Reciprocal Rank Fusion (RRF)
   - Merges results from BM25 and vector search
   - Weighted combination with configurable weights
   - Industry standard (used by Elasticsearch, Weaviate)

4. Cross-Encoder Reranking
   - Final reranking with joint query-document encoding
   - More accurate than bi-encoder similarity
   - Computes precise relevance scores

This approach outperforms single-method retrieval by 20-30% in most cases.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .cross_encoder import CrossEncoderReranker, get_reranker

logger = structlog.get_logger(__name__)


class RetrievalSource(Enum):
    """Source of a retrieval result."""

    BM25_ONLY = "bm25_only"
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    RERANKED = "reranked"


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata.

    Attributes:
        id: Document ID
        content: Document text/content
        metadata: Additional metadata (error_type, selector, etc.)
        bm25_score: BM25 keyword score (0.0 if not from BM25)
        vector_score: Vector similarity score (0.0 if not from vector)
        hybrid_score: Combined RRF score
        rerank_score: Cross-encoder reranking score (if reranked)
        final_score: Final score used for ranking
        source: Which retrieval method produced this result
        rank: Final rank position (1-indexed)
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    bm25_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    source: RetrievalSource = RetrievalSource.HYBRID
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "bm25_score": self.bm25_score,
            "vector_score": self.vector_score,
            "hybrid_score": self.hybrid_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "source": self.source.value,
            "rank": self.rank,
        }


class HybridRetriever:
    """Hybrid retriever combining BM25, vector search, and reranking.

    This retriever implements the complete hybrid search pipeline:
    1. BM25 keyword search
    2. Vector semantic search
    3. Reciprocal Rank Fusion (RRF) merging
    4. Cross-encoder reranking

    Example:
        ```python
        retriever = HybridRetriever(database_url="postgresql://...")

        results = await retriever.retrieve(
            query="Element not found: button#submit",
            limit=5,
            enable_reranking=True
        )

        for result in results:
            print(f"{result.rank}. {result.content} (score: {result.final_score:.3f})")
        ```
    """

    def __init__(
        self,
        database_url: str | None = None,
        embeddings: Any | None = None,
        reranker: CrossEncoderReranker | None = None,
        enable_reranking: bool = True,
    ):
        """Initialize the hybrid retriever.

        Args:
            database_url: PostgreSQL connection URL
            embeddings: LangChain embeddings instance for vector search
            reranker: Cross-encoder reranker instance
            enable_reranking: Whether to enable cross-encoder reranking
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.embeddings = embeddings
        self.enable_reranking = enable_reranking
        self._pool = None
        self._log = logger.bind(component="hybrid_retriever")

        # Initialize reranker
        if enable_reranking:
            self.reranker = reranker or get_reranker()
        else:
            self.reranker = None

    async def _get_pool(self):
        """Get or create the database connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for hybrid retriever. Install with: pip install asyncpg"
                )

            if not self.database_url:
                raise ValueError("DATABASE_URL environment variable or database_url parameter required")

            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
            )
            self._log.info("Created database connection pool")

        return self._pool

    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._log.info("Closed database connection pool")

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if embeddings not configured
        """
        if not self.embeddings:
            return None

        try:
            # Handle both sync and async embeddings
            if hasattr(self.embeddings, "aembed_query"):
                return await self.embeddings.aembed_query(text)
            elif hasattr(self.embeddings, "embed_query"):
                return self.embeddings.embed_query(text)
            else:
                self._log.warning("Embeddings instance has no embed_query method")
                return None
        except Exception as e:
            self._log.error("Failed to generate embedding", error=str(e))
            return None

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.01,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        error_type_filter: str | None = None,
        enable_reranking: bool | None = None,
        rerank_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant failure patterns using hybrid search.

        Args:
            query: Search query (error message, selector, etc.)
            limit: Maximum number of results to return
            threshold: Minimum hybrid score threshold (0.0-1.0)
            bm25_weight: Weight for BM25 scores in RRF (0.0-1.0)
            vector_weight: Weight for vector scores in RRF (0.0-1.0)
            error_type_filter: Optional filter by error type
            enable_reranking: Override instance-level reranking setting
            rerank_top_k: Number of results to rerank (defaults to limit * 2)

        Returns:
            List of RetrievalResult sorted by relevance
        """
        # Generate embedding for vector search
        query_embedding = await self._get_embedding(query)

        if not query_embedding:
            self._log.warning("No embedding available, falling back to BM25 only")
            return await self._retrieve_bm25_only(query, limit, error_type_filter)

        # Get database pool
        pool = await self._get_pool()

        # Convert embedding to string format for pgvector
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        # Normalize weights
        total_weight = bm25_weight + vector_weight
        if total_weight > 0:
            bm25_weight = bm25_weight / total_weight
            vector_weight = vector_weight / total_weight

        # Execute hybrid search query
        # We retrieve more results than needed for reranking
        fetch_limit = rerank_top_k or (limit * 2) if self.enable_reranking else limit

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM search_failure_patterns_hybrid(
                    $1::TEXT,
                    $2::vector(1536),
                    $3::FLOAT,
                    $4::INT,
                    $5::TEXT,
                    $6::FLOAT,
                    $7::FLOAT
                )
                """,
                query,
                embedding_str,
                threshold,
                fetch_limit,
                error_type_filter,
                bm25_weight,
                vector_weight,
            )

        # Convert to RetrievalResult objects
        results = []
        for row in rows:
            # Extract content from row
            content = self._build_content_from_row(row)

            result = RetrievalResult(
                id=str(row["id"]),
                content=content,
                metadata={
                    "error_message": row["error_message"],
                    "error_type": row["error_type"],
                    "selector": row["selector"],
                    "healed_selector": row["healed_selector"],
                    "healing_method": row["healing_method"],
                    "success_count": row["success_count"],
                    "failure_count": row["failure_count"],
                    "success_rate": float(row["success_rate"]),
                },
                bm25_score=float(row["bm25_score"]),
                vector_score=float(row["vector_score"]),
                hybrid_score=float(row["hybrid_score"]),
                final_score=float(row["hybrid_score"]),
                source=RetrievalSource.HYBRID,
            )
            results.append(result)

        # Apply reranking if enabled
        use_reranking = enable_reranking if enable_reranking is not None else self.enable_reranking
        if use_reranking and self.reranker and len(results) > 1:
            results = await self._rerank_results(query, results, limit)

        # Apply final limit and set ranks
        results = results[:limit]
        for i, result in enumerate(results, 1):
            result.rank = i

        self._log.info(
            "Hybrid retrieval completed",
            query_length=len(query),
            num_results=len(results),
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            reranked=use_reranking,
        )

        return results

    async def _retrieve_bm25_only(
        self,
        query: str,
        limit: int,
        error_type_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Fallback to BM25-only retrieval when embeddings unavailable.

        Args:
            query: Search query
            limit: Maximum number of results
            error_type_filter: Optional error type filter

        Returns:
            List of RetrievalResult from BM25 search
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM search_failure_patterns_bm25(
                    $1::TEXT,
                    $2::FLOAT,
                    $3::INT,
                    $4::TEXT
                )
                """,
                query,
                0.0,  # threshold
                limit,
                error_type_filter,
            )

        results = []
        for i, row in enumerate(rows, 1):
            content = self._build_content_from_row(row)

            result = RetrievalResult(
                id=str(row["id"]),
                content=content,
                metadata={
                    "error_message": row["error_message"],
                    "error_type": row["error_type"],
                    "selector": row["selector"],
                    "healed_selector": row["healed_selector"],
                    "healing_method": row["healing_method"],
                    "success_count": row["success_count"],
                    "failure_count": row["failure_count"],
                    "success_rate": float(row["success_rate"]),
                },
                bm25_score=float(row["bm25_rank"]),
                final_score=float(row["bm25_rank"]),
                source=RetrievalSource.BM25_ONLY,
                rank=i,
            )
            results.append(result)

        self._log.info(
            "BM25-only retrieval completed",
            query_length=len(query),
            num_results=len(results),
        )

        return results

    def _build_content_from_row(self, row: dict) -> str:
        """Build content string from database row.

        Args:
            row: Database row as dict

        Returns:
            Formatted content string for reranking
        """
        parts = []

        if row.get("error_message"):
            parts.append(f"Error: {row['error_message']}")

        if row.get("selector"):
            parts.append(f"Selector: {row['selector']}")

        if row.get("healed_selector"):
            parts.append(f"Fixed: {row['healed_selector']}")

        if row.get("healing_method"):
            parts.append(f"Method: {row['healing_method']}")

        return " | ".join(parts)

    async def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder.

        Args:
            query: Original search query
            results: Initial retrieval results
            top_k: Number of top results to keep after reranking

        Returns:
            Reranked results sorted by rerank_score
        """
        if not self.reranker:
            return results

        # Extract document texts for reranking
        documents = [r.content for r in results]

        # Rerank
        reranked = await self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=None,  # Rerank all, then apply top_k
            return_scores=True,
        )

        # Update results with rerank scores
        reranked_results = []
        for rerank_item in reranked:
            original_index = rerank_item["index"]
            result = results[original_index]

            result.rerank_score = rerank_item["score"]
            result.final_score = rerank_item["score"]
            result.source = RetrievalSource.RERANKED

            reranked_results.append(result)

        self._log.debug(
            "Reranked results",
            num_results=len(reranked_results),
            top_k=top_k,
        )

        return reranked_results[:top_k]

    async def get_retrieval_stats(self) -> dict[str, Any]:
        """Get retrieval system statistics.

        Returns:
            Dictionary with stats about the retrieval system
        """
        stats = {
            "embeddings_available": self.embeddings is not None,
            "reranking_enabled": self.enable_reranking,
        }

        if self.reranker:
            stats["reranker_cache"] = self.reranker.get_cache_stats()

        # Get database stats
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Count patterns
                count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM test_failure_patterns
                    WHERE embedding IS NOT NULL
                    """
                )
                stats["indexed_patterns"] = count

                # Count patterns with FTS
                fts_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM test_failure_patterns
                    WHERE error_message_tsv IS NOT NULL
                    """
                )
                stats["fts_indexed_patterns"] = fts_count

        except Exception as e:
            self._log.warning("Failed to get database stats", error=str(e))
            stats["error"] = str(e)

        return stats


# ============================================================================
# Global Instance Management
# ============================================================================

_hybrid_retriever: HybridRetriever | None = None


def get_hybrid_retriever(
    database_url: str | None = None,
    embeddings: Any | None = None,
    reranker: CrossEncoderReranker | None = None,
    enable_reranking: bool = True,
) -> HybridRetriever:
    """Get or create the global hybrid retriever instance.

    Args:
        database_url: Database URL (only used on first call)
        embeddings: Embeddings instance (only used on first call)
        reranker: Reranker instance (only used on first call)
        enable_reranking: Enable reranking (only used on first call)

    Returns:
        HybridRetriever instance
    """
    global _hybrid_retriever

    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(
            database_url=database_url,
            embeddings=embeddings,
            reranker=reranker,
            enable_reranking=enable_reranking,
        )

    return _hybrid_retriever


def reset_hybrid_retriever() -> None:
    """Reset the global hybrid retriever instance.

    Useful for testing or when configuration changes.
    """
    global _hybrid_retriever
    _hybrid_retriever = None


async def init_hybrid_retriever(
    database_url: str | None = None,
    embeddings: Any | None = None,
    reranker: CrossEncoderReranker | None = None,
    enable_reranking: bool = True,
) -> HybridRetriever:
    """Initialize and verify the hybrid retriever.

    Args:
        database_url: Database URL
        embeddings: Embeddings instance
        reranker: Reranker instance
        enable_reranking: Enable reranking

    Returns:
        Initialized HybridRetriever instance
    """
    retriever = get_hybrid_retriever(
        database_url=database_url,
        embeddings=embeddings,
        reranker=reranker,
        enable_reranking=enable_reranking,
    )

    # Verify connection
    await retriever._get_pool()

    logger.info(
        "Hybrid retriever initialized successfully",
        embeddings_available=retriever.embeddings is not None,
        reranking_enabled=retriever.enable_reranking,
    )

    return retriever
