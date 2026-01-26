"""Cross-encoder reranking for hybrid search.

Cross-encoders provide more accurate relevance scoring than bi-encoders
by jointly encoding the query and document. This is used as a final
reranking step after initial retrieval.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking dataset
- Fast inference (6 layers, ~80MB model)
- Good accuracy for general-purpose reranking
"""

import hashlib
import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Lazy import to avoid loading model if not needed
_sentence_transformers = None
_CrossEncoder = None


def _import_sentence_transformers():
    """Lazy import of sentence-transformers."""
    global _sentence_transformers, _CrossEncoder
    if _sentence_transformers is None:
        try:
            import sentence_transformers

            _sentence_transformers = sentence_transformers
            _CrossEncoder = sentence_transformers.CrossEncoder
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            _sentence_transformers = False
            _CrossEncoder = None
    return _CrossEncoder


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval quality.

    Usage:
        ```python
        reranker = CrossEncoderReranker()
        results = await reranker.rerank(
            query="Element not found error",
            documents=["Button missing", "Login failed", "Timeout error"],
            top_k=3
        )
        # Returns documents sorted by relevance score
        ```

    The reranker uses an LRU cache to avoid recomputing scores for
    frequently seen query-document pairs.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CACHE_SIZE = 1000

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_enabled: bool = True,
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Model name from HuggingFace. Defaults to ms-marco-MiniLM-L-6-v2.
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            cache_enabled: Whether to cache reranking scores.
        """
        CrossEncoder = _import_sentence_transformers()

        if CrossEncoder is None:
            # sentence-transformers not available
            self.model = None
            self.cache_enabled = False
            logger.warning("Cross-encoder reranker disabled (sentence-transformers not installed)")
            return

        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_enabled = cache_enabled
        self._cache: dict[str, float] = {}

        # Auto-detect device if not specified
        if device is None:
            device = self._auto_detect_device()

        try:
            self.model = CrossEncoder(self.model_name, device=device)
            logger.info(
                "Initialized cross-encoder reranker",
                model=self.model_name,
                device=device,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize cross-encoder",
                model=self.model_name,
                error=str(e),
            )
            self.model = None

    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            # Apple Silicon
            return "mps"
        else:
            return "cpu"

    def _get_cache_key(self, query: str, document: str) -> str:
        """Generate cache key for query-document pair."""
        key = f"{query}||{document}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cached_score(self, query: str, document: str) -> float | None:
        """Get cached score if available."""
        if not self.cache_enabled:
            return None

        cache_key = self._get_cache_key(query, document)
        return self._cache.get(cache_key)

    def _cache_score(self, query: str, document: str, score: float) -> None:
        """Cache a score."""
        if not self.cache_enabled:
            return

        # Simple LRU: if cache is full, clear it
        if len(self._cache) >= self.CACHE_SIZE:
            self._cache.clear()
            logger.debug("Cleared reranker cache", size=self.CACHE_SIZE)

        cache_key = self._get_cache_key(query, document)
        self._cache[cache_key] = score

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        return_scores: bool = True,
    ) -> list[dict[str, Any]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return. None = all results.
            return_scores: Whether to include scores in results

        Returns:
            List of dicts with 'text', 'score', 'index' (original position)
        """
        if not self.model or not documents:
            # If model not available or no documents, return in original order
            return [
                {"text": doc, "score": 0.0, "index": i}
                for i, doc in enumerate(documents[:top_k] if top_k else documents)
            ]

        # Check cache first
        results = []
        uncached_indices = []
        uncached_pairs = []

        for i, doc in enumerate(documents):
            cached_score = self._get_cached_score(query, doc)
            if cached_score is not None:
                results.append({"text": doc, "score": cached_score, "index": i})
            else:
                uncached_indices.append(i)
                uncached_pairs.append([query, doc])

        # Score uncached pairs
        if uncached_pairs:
            try:
                scores = self.model.predict(uncached_pairs, convert_to_numpy=True)

                for idx, score in zip(uncached_indices, scores):
                    doc = documents[idx]
                    score_float = float(score)
                    results.append({"text": doc, "score": score_float, "index": idx})
                    self._cache_score(query, doc, score_float)

            except Exception as e:
                logger.error("Reranking failed", error=str(e))
                # Fallback: return documents in original order with zero scores
                for idx in uncached_indices:
                    results.append({"text": documents[idx], "score": 0.0, "index": idx})

        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Apply top_k
        if top_k is not None:
            results = results[:top_k]

        if not return_scores:
            for r in results:
                del r["score"]

        logger.debug(
            "Reranked documents",
            query_length=len(query),
            num_documents=len(documents),
            num_cached=len(documents) - len(uncached_pairs),
            top_k=top_k or len(results),
        )

        return results

    def clear_cache(self) -> None:
        """Clear the reranking cache."""
        self._cache.clear()
        logger.info("Cleared cross-encoder cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.cache_enabled,
            "size": len(self._cache),
            "max_size": self.CACHE_SIZE,
            "hit_rate": None,  # Would need to track hits/misses
        }


# ============================================================================
# Global Instance Management
# ============================================================================

_reranker: CrossEncoderReranker | None = None


def get_reranker(
    model_name: str | None = None,
    device: str | None = None,
    cache_enabled: bool = True,
) -> CrossEncoderReranker:
    """Get or create the global reranker instance.

    Args:
        model_name: Model name (only used on first call)
        device: Device (only used on first call)
        cache_enabled: Enable caching (only used on first call)

    Returns:
        CrossEncoderReranker instance
    """
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model_name=model_name,
            device=device,
            cache_enabled=cache_enabled,
        )

    return _reranker


def reset_reranker() -> None:
    """Reset the global reranker instance.

    Useful for testing or when configuration changes.
    """
    global _reranker
    _reranker = None
