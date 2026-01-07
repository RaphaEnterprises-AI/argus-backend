"""Local Embedder - Privacy-preserving code embeddings.

KEY INNOVATION: Code NEVER leaves the user's machine.
We generate embeddings locally using sentence-transformers.

Supported models (in order of quality):
1. BAAI/bge-large-en-v1.5 (1.3GB) - Best quality
2. BAAI/bge-base-en-v1.5 (420MB) - Good balance
3. BAAI/bge-small-en-v1.5 (130MB) - Fast, small
4. all-MiniLM-L6-v2 (90MB) - Fallback

Only the embeddings (vectors) are sent to Cloudflare Vectorize,
never the actual code content.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np

from .semantic_chunker import CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding a code chunk."""

    chunk_id: str
    embedding: list[float]
    dimension: int
    model_name: str

    # Metadata
    token_count: int = 0
    embed_time_ms: float = 0.0

    # For caching
    content_hash: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "token_count": self.token_count,
            "embed_time_ms": self.embed_time_ms,
            "content_hash": self.content_hash,
        }


@dataclass
class EmbeddingConfig:
    """Configuration for local embedder."""

    # Model selection
    model_name: str = "BAAI/bge-base-en-v1.5"  # Default: good balance

    # Performance
    batch_size: int = 32        # Chunks to embed at once
    max_length: int = 512       # Max tokens per chunk
    use_gpu: bool = True        # Use GPU if available
    normalize: bool = True      # L2 normalize embeddings

    # Caching
    cache_embeddings: bool = True
    cache_dir: str = ".argus/embeddings"

    # Fallback
    fallback_model: str = "all-MiniLM-L6-v2"


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding."""

    results: list[EmbeddingResult]
    total_chunks: int
    successful: int
    failed: int
    total_time_ms: float
    average_time_ms: float


class LocalEmbedder:
    """Privacy-preserving local code embeddings.

    Usage:
        embedder = LocalEmbedder()
        chunk = CodeChunk(...)
        result = embedder.embed(chunk.to_embedding_text())

        # Batch processing
        results = embedder.embed_batch([chunk1, chunk2, chunk3])
    """

    # Model info for different embedding models
    MODEL_INFO = {
        "BAAI/bge-large-en-v1.5": {"dim": 1024, "size_mb": 1300},
        "BAAI/bge-base-en-v1.5": {"dim": 768, "size_mb": 420},
        "BAAI/bge-small-en-v1.5": {"dim": 384, "size_mb": 130},
        "all-MiniLM-L6-v2": {"dim": 384, "size_mb": 90},
        "microsoft/codebert-base": {"dim": 768, "size_mb": 440},
        "jinaai/jina-embeddings-v2-base-code": {"dim": 768, "size_mb": 550},
    }

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedder with optional configuration."""
        self.config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._device = "cpu"

        # Embedding cache
        self._cache: dict[str, list[float]] = {}

    def _initialize(self) -> bool:
        """Lazy initialization of the embedding model."""
        if self._initialized:
            return True

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Determine device
            if self.config.use_gpu and torch.cuda.is_available():
                self._device = "cuda"
            elif self.config.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"  # Apple Silicon
            else:
                self._device = "cpu"

            logger.info(f"Loading embedding model: {self.config.model_name} on {self._device}")

            try:
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self._device
                )
            except Exception as e:
                logger.warning(f"Failed to load {self.config.model_name}: {e}")
                logger.info(f"Falling back to {self.config.fallback_model}")
                self._model = SentenceTransformer(
                    self.config.fallback_model,
                    device=self._device
                )
                self.config.model_name = self.config.fallback_model

            self._initialized = True
            logger.info(f"Embedder initialized: {self.config.model_name} ({self._device})")
            return True

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return False

    @property
    def dimension(self) -> int:
        """Get embedding dimension for current model."""
        return self.MODEL_INFO.get(
            self.config.model_name,
            {"dim": 384}
        )["dim"]

    @property
    def is_available(self) -> bool:
        """Check if embedder is available."""
        return self._initialize()

    def embed(
        self,
        text: str,
        chunk_id: Optional[str] = None
    ) -> Optional[EmbeddingResult]:
        """Embed a single text string.

        Args:
            text: Text to embed (usually from chunk.to_embedding_text())
            chunk_id: Optional chunk ID for tracking

        Returns:
            EmbeddingResult or None if embedding failed
        """
        if not self._initialize():
            return None

        # Check cache
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        if self.config.cache_embeddings and content_hash in self._cache:
            return EmbeddingResult(
                chunk_id=chunk_id or content_hash,
                embedding=self._cache[content_hash],
                dimension=len(self._cache[content_hash]),
                model_name=self.config.model_name,
                content_hash=content_hash,
            )

        try:
            start = time.perf_counter()

            # Generate embedding
            embedding = self._model.encode(
                text,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000

            # Convert to list
            embedding_list = embedding.tolist()

            # Cache
            if self.config.cache_embeddings:
                self._cache[content_hash] = embedding_list

            return EmbeddingResult(
                chunk_id=chunk_id or content_hash,
                embedding=embedding_list,
                dimension=len(embedding_list),
                model_name=self.config.model_name,
                token_count=len(text.split()),  # Rough estimate
                embed_time_ms=elapsed_ms,
                content_hash=content_hash,
            )

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def embed_chunk(self, chunk: CodeChunk) -> Optional[EmbeddingResult]:
        """Embed a code chunk.

        Args:
            chunk: CodeChunk to embed

        Returns:
            EmbeddingResult or None if embedding failed
        """
        text = chunk.to_embedding_text()
        return self.embed(text, chunk_id=chunk.chunk_id)

    def embed_batch(
        self,
        chunks: list[CodeChunk],
        show_progress: bool = False
    ) -> BatchEmbeddingResult:
        """Embed multiple chunks in batch for efficiency.

        Args:
            chunks: List of CodeChunks to embed
            show_progress: Show progress bar

        Returns:
            BatchEmbeddingResult with all results
        """
        if not self._initialize():
            return BatchEmbeddingResult(
                results=[],
                total_chunks=len(chunks),
                successful=0,
                failed=len(chunks),
                total_time_ms=0,
                average_time_ms=0,
            )

        start = time.perf_counter()
        results = []
        failed = 0

        # Prepare texts
        texts = [chunk.to_embedding_text() for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        content_hashes = [
            hashlib.sha256(t.encode()).hexdigest()[:16]
            for t in texts
        ]

        # Check cache for hits
        cache_hits = []
        texts_to_embed = []
        indices_to_embed = []

        for i, (text, hash_val) in enumerate(zip(texts, content_hashes)):
            if self.config.cache_embeddings and hash_val in self._cache:
                cache_hits.append((i, self._cache[hash_val]))
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Batch embed non-cached texts
        if texts_to_embed:
            try:
                embeddings = self._model.encode(
                    texts_to_embed,
                    normalize_embeddings=self.config.normalize,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                    batch_size=self.config.batch_size,
                )

                # Store results and cache
                for i, (original_idx, embedding) in enumerate(zip(indices_to_embed, embeddings)):
                    embedding_list = embedding.tolist()

                    # Cache
                    if self.config.cache_embeddings:
                        self._cache[content_hashes[original_idx]] = embedding_list

                    results.append((original_idx, EmbeddingResult(
                        chunk_id=chunk_ids[original_idx],
                        embedding=embedding_list,
                        dimension=len(embedding_list),
                        model_name=self.config.model_name,
                        content_hash=content_hashes[original_idx],
                    )))

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                failed = len(texts_to_embed)

        # Add cache hits
        for original_idx, cached_embedding in cache_hits:
            results.append((original_idx, EmbeddingResult(
                chunk_id=chunk_ids[original_idx],
                embedding=cached_embedding,
                dimension=len(cached_embedding),
                model_name=self.config.model_name,
                content_hash=content_hashes[original_idx],
            )))

        # Sort by original index and extract results
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]

        elapsed_ms = (time.perf_counter() - start) * 1000
        successful = len(final_results)

        return BatchEmbeddingResult(
            results=final_results,
            total_chunks=len(chunks),
            successful=successful,
            failed=failed,
            total_time_ms=elapsed_ms,
            average_time_ms=elapsed_ms / max(successful, 1),
        )

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = False
    ) -> list[Optional[list[float]]]:
        """Embed multiple text strings directly.

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            List of embeddings (None for failed)
        """
        if not self._initialize():
            return [None] * len(texts)

        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                batch_size=self.config.batch_size,
            )

            return [e.tolist() for e in embeddings]

        except Exception as e:
            logger.error(f"Batch text embedding failed: {e}")
            return [None] * len(texts)

    def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def find_similar(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[tuple[str, list[float]]],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        """Find most similar embeddings to a query.

        Args:
            query_embedding: Query vector
            candidate_embeddings: List of (id, embedding) tuples
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (id, similarity) tuples, sorted by similarity
        """
        results = []

        for chunk_id, embedding in candidate_embeddings:
            similarity = self.compute_similarity(query_embedding, embedding)
            if similarity >= threshold:
                results.append((chunk_id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def save_cache(self, path: Optional[str] = None) -> bool:
        """Save embedding cache to disk.

        Args:
            path: Path to save cache (default: config.cache_dir)

        Returns:
            True if successful
        """
        import json

        try:
            cache_path = Path(path or self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            cache_file = cache_path / "embeddings_cache.json"

            with open(cache_file, "w") as f:
                json.dump({
                    "model": self.config.model_name,
                    "embeddings": self._cache,
                }, f)

            logger.info(f"Saved {len(self._cache)} embeddings to {cache_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False

    def load_cache(self, path: Optional[str] = None) -> bool:
        """Load embedding cache from disk.

        Args:
            path: Path to load cache from (default: config.cache_dir)

        Returns:
            True if successful
        """
        import json

        try:
            cache_path = Path(path or self.config.cache_dir)
            cache_file = cache_path / "embeddings_cache.json"

            if not cache_file.exists():
                return False

            with open(cache_file) as f:
                data = json.load(f)

            # Only load if same model
            if data.get("model") != self.config.model_name:
                logger.warning(
                    f"Cache model mismatch: {data.get('model')} vs {self.config.model_name}"
                )
                return False

            self._cache = data.get("embeddings", {})
            logger.info(f"Loaded {len(self._cache)} embeddings from cache")
            return True

        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache = {}
        logger.info("Embedding cache cleared")

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        info = self.MODEL_INFO.get(self.config.model_name, {})
        return {
            "model_name": self.config.model_name,
            "dimension": info.get("dim", self.dimension),
            "size_mb": info.get("size_mb", 0),
            "device": self._device,
            "initialized": self._initialized,
            "cache_size": len(self._cache),
        }


# Singleton for code embedding
class CodeEmbedder(LocalEmbedder):
    """Specialized embedder for code with code-specific prompts.

    BGE models benefit from task-specific prefixes:
    - "Represent the code for retrieval: "
    """

    CODE_PREFIX = "Represent the code for retrieval: "
    QUERY_PREFIX = "Represent the query for retrieving code: "

    def embed(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        is_query: bool = False
    ) -> Optional[EmbeddingResult]:
        """Embed code with appropriate prefix.

        Args:
            text: Code or query text
            chunk_id: Chunk identifier
            is_query: Whether this is a search query (not code)

        Returns:
            EmbeddingResult
        """
        # Add prefix for BGE models
        if "bge" in self.config.model_name.lower():
            prefix = self.QUERY_PREFIX if is_query else self.CODE_PREFIX
            text = prefix + text

        return super().embed(text, chunk_id)

    def embed_query(self, query: str) -> Optional[list[float]]:
        """Embed a search query.

        Args:
            query: Search query text

        Returns:
            Embedding vector or None
        """
        result = self.embed(query, is_query=True)
        return result.embedding if result else None


# Global instances
_embedder: Optional[LocalEmbedder] = None
_code_embedder: Optional[CodeEmbedder] = None


def get_embedder(config: Optional[EmbeddingConfig] = None) -> LocalEmbedder:
    """Get or create global embedder instance."""
    global _embedder
    if _embedder is None or config is not None:
        _embedder = LocalEmbedder(config)
    return _embedder


def get_code_embedder(config: Optional[EmbeddingConfig] = None) -> CodeEmbedder:
    """Get or create global code embedder instance."""
    global _code_embedder
    if _code_embedder is None or config is not None:
        _code_embedder = CodeEmbedder(config)
    return _code_embedder
