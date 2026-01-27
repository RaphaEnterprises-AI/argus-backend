"""Long-term memory store for cross-session learning.

.. deprecated:: 2026.01
    This module is deprecated. Use `src.knowledge.CogneeKnowledgeClient` instead.
    The Cognee-based client provides:
    - Same API (drop-in replacement)
    - Better knowledge extraction via ECL pipeline
    - Graph + vector hybrid search
    - Multi-hop reasoning
    - 30+ data connectors

    Migration:
        ```python
        # Old way (deprecated)
        from src.orchestrator.memory_store import get_memory_store
        store = get_memory_store()

        # New way
        from src.knowledge import get_cognee_client
        client = get_cognee_client(org_id="...", project_id="...")
        ```

    See RAP-132 for migration details.

This module provides a persistent memory store for LangGraph that enables:
- Cross-session learning from test failures
- Semantic search on failure patterns using pgvector
- Healing pattern storage and retrieval
- Success/failure tracking for continuous improvement

The memory store uses Supabase PostgreSQL with pgvector for semantic search,
allowing the system to find similar past failures and apply proven healing solutions.
"""

import warnings

warnings.warn(
    "memory_store is deprecated. Use src.knowledge.CogneeKnowledgeClient instead. "
    "See RAP-132 for migration details.",
    DeprecationWarning,
    stacklevel=2,
)

import json
import os
import uuid
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MemoryStore:
    """Memory store for long-term learning with semantic search.

    This store uses PostgreSQL with pgvector to enable:
    - Storing arbitrary key-value data with optional embeddings
    - Semantic search across stored memories
    - Specialized failure pattern storage and retrieval
    - Success/failure tracking for healing patterns

    Example usage:
        ```python
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
        store = MemoryStore(embeddings=embeddings)

        # Store a healing pattern
        await store.store_failure_pattern(
            error_message="Element not found: #submit-btn",
            healed_selector="#submit-button",
            healing_method="semantic_match",
        )

        # Find similar failures
        similar = await store.find_similar_failures(
            error_message="Element not found: #submit"
        )
        ```
    """

    def __init__(
        self,
        database_url: str | None = None,
        embeddings: Any | None = None,  # Embeddings instance
    ):
        """Initialize the memory store.

        Args:
            database_url: PostgreSQL connection URL. Defaults to DATABASE_URL env var.
            embeddings: LangChain embeddings instance for semantic search.
                       If not provided, semantic search will be disabled.
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.embeddings = embeddings
        self._pool = None
        self._log = logger.bind(component="memory_store")

    async def _get_pool(self):
        """Get or create the connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for memory store. Install with: pip install asyncpg"
                )

            if not self.database_url:
                raise ValueError(
                    "DATABASE_URL environment variable or database_url parameter required"
                )

            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
            )
            self._log.info("Created database connection pool")

        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._log.info("Closed database connection pool")

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text using configured embeddings model.

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

    # =========================================================================
    # General Memory Store Operations
    # =========================================================================

    async def put(
        self,
        namespace: list[str],
        key: str,
        value: dict[str, Any],
        embed_text: str | None = None,
    ) -> None:
        """Store a value with optional embedding.

        Args:
            namespace: Hierarchical namespace for the key (e.g., ["tests", "login"])
            key: Unique key within the namespace
            value: JSON-serializable value to store
            embed_text: Optional text to generate embedding for semantic search
        """
        pool = await self._get_pool()

        embedding = None
        if embed_text:
            embedding = await self._get_embedding(embed_text)

        async with pool.acquire() as conn:
            # Convert embedding to string format for pgvector
            embedding_str = None
            if embedding:
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"

            await conn.execute(
                """
                INSERT INTO langgraph_memory_store (namespace, key, value, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (namespace, key)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                """,
                namespace,
                key,
                json.dumps(value),
                embedding_str,
            )

        self._log.debug("Stored memory", namespace=namespace, key=key)

    async def get(
        self,
        namespace: list[str],
        key: str,
    ) -> dict[str, Any] | None:
        """Get a value by namespace and key.

        Args:
            namespace: Hierarchical namespace
            key: Key within the namespace

        Returns:
            Stored value or None if not found
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT value FROM langgraph_memory_store
                WHERE namespace = $1 AND key = $2
                """,
                namespace,
                key,
            )

            if row:
                return json.loads(row["value"])
            return None

    async def delete(
        self,
        namespace: list[str],
        key: str,
    ) -> bool:
        """Delete a value by namespace and key.

        Args:
            namespace: Hierarchical namespace
            key: Key within the namespace

        Returns:
            True if deleted, False if not found
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM langgraph_memory_store
                WHERE namespace = $1 AND key = $2
                """,
                namespace,
                key,
            )

            deleted = result.split()[-1] != "0"
            if deleted:
                self._log.debug("Deleted memory", namespace=namespace, key=key)
            return deleted

    async def list_keys(
        self,
        namespace: list[str],
        limit: int = 100,
    ) -> list[str]:
        """List all keys in a namespace.

        Args:
            namespace: Hierarchical namespace
            limit: Maximum number of keys to return

        Returns:
            List of keys in the namespace
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT key FROM langgraph_memory_store
                WHERE namespace = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                namespace,
                limit,
            )

            return [row["key"] for row in rows]

    async def search(
        self,
        namespace: list[str],
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Semantic search within a namespace.

        Args:
            namespace: Hierarchical namespace to search in
            query: Search query text
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of matching items with similarity scores
        """
        if not self.embeddings:
            self._log.warning("No embeddings configured for semantic search")
            return []

        query_embedding = await self._get_embedding(query)
        if not query_embedding:
            return []

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Convert embedding to string format for pgvector
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            rows = await conn.fetch(
                """
                SELECT key, value, 1 - (embedding <=> $1::vector) as similarity
                FROM langgraph_memory_store
                WHERE namespace = $2
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> $1::vector) > $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                embedding_str,
                namespace,
                threshold,
                limit,
            )

            return [
                {
                    "key": row["key"],
                    "value": json.loads(row["value"]),
                    "similarity": float(row["similarity"]),
                }
                for row in rows
            ]

    # =========================================================================
    # Failure Pattern Storage (Specialized for Self-Healing)
    # =========================================================================

    async def store_failure_pattern(
        self,
        error_message: str,
        healed_selector: str,
        healing_method: str,
        test_id: str | None = None,
        error_type: str | None = None,
        original_selector: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a test failure pattern for future learning.

        This stores a successful healing pattern that can be retrieved
        when similar failures occur in the future.

        Args:
            error_message: The error message from the test failure
            healed_selector: The selector that fixed the issue
            healing_method: Method used to heal (e.g., "semantic_match", "code_aware")
            test_id: Optional test ID for correlation
            error_type: Type of error (e.g., "selector_changed", "timing_issue")
            original_selector: The original broken selector
            metadata: Additional metadata to store

        Returns:
            ID of the stored pattern
        """
        pool = await self._get_pool()

        # Generate embedding from error message for semantic search
        embedding = await self._get_embedding(error_message)

        # Parse test_id as UUID if provided
        test_uuid = None
        if test_id:
            try:
                test_uuid = uuid.UUID(test_id)
            except ValueError:
                # If not a valid UUID, create one from the string
                test_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, test_id)

        async with pool.acquire() as conn:
            # Convert embedding to string format for pgvector
            embedding_str = None
            if embedding:
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"

            row = await conn.fetchrow(
                """
                INSERT INTO test_failure_patterns
                (test_id, error_message, error_type, selector, healed_selector,
                 healing_method, embedding, metadata, success_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, 1)
                RETURNING id
                """,
                test_uuid,
                error_message,
                error_type,
                original_selector,
                healed_selector,
                healing_method,
                embedding_str,
                json.dumps(metadata or {}),
            )

            pattern_id = str(row["id"])

            self._log.info(
                "Stored failure pattern",
                pattern_id=pattern_id,
                error_type=error_type,
                healing_method=healing_method,
            )

            return pattern_id

    async def find_similar_failures(
        self,
        error_message: str,
        limit: int = 5,
        threshold: float = 0.7,
        error_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find similar past failures with their healing solutions.

        Uses semantic search to find failures with similar error messages
        and returns the healing solutions that worked.

        Args:
            error_message: The current error message to match
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)
            error_type: Optional filter by error type

        Returns:
            List of similar failures with healing solutions and success rates
        """
        if not self.embeddings:
            self._log.warning("No embeddings configured for semantic search")
            return []

        query_embedding = await self._get_embedding(error_message)
        if not query_embedding:
            return []

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Convert embedding to string format for pgvector
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            # Use the database function for optimized search
            if error_type:
                rows = await conn.fetch(
                    """
                    SELECT
                        id,
                        error_message,
                        error_type,
                        selector,
                        healed_selector,
                        healing_method,
                        success_count,
                        failure_count,
                        metadata,
                        1 - (embedding <=> $1::vector) AS similarity,
                        CASE
                            WHEN (success_count + failure_count) > 0
                            THEN success_count::FLOAT / (success_count + failure_count)
                            ELSE 0
                        END AS success_rate
                    FROM test_failure_patterns
                    WHERE embedding IS NOT NULL
                      AND error_type = $4
                      AND 1 - (embedding <=> $1::vector) > $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    embedding_str,
                    threshold,
                    limit,
                    error_type,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        id,
                        error_message,
                        error_type,
                        selector,
                        healed_selector,
                        healing_method,
                        success_count,
                        failure_count,
                        metadata,
                        1 - (embedding <=> $1::vector) AS similarity,
                        CASE
                            WHEN (success_count + failure_count) > 0
                            THEN success_count::FLOAT / (success_count + failure_count)
                            ELSE 0
                        END AS success_rate
                    FROM test_failure_patterns
                    WHERE embedding IS NOT NULL
                      AND 1 - (embedding <=> $1::vector) > $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    embedding_str,
                    threshold,
                    limit,
                )

            results = []
            for row in rows:
                results.append({
                    "id": str(row["id"]),
                    "error_message": row["error_message"],
                    "error_type": row["error_type"],
                    "original_selector": row["selector"],
                    "healed_selector": row["healed_selector"],
                    "healing_method": row["healing_method"],
                    "success_count": row["success_count"],
                    "failure_count": row["failure_count"],
                    "success_rate": float(row["success_rate"]),
                    "similarity": float(row["similarity"]),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                })

            self._log.debug(
                "Found similar failures",
                query=error_message[:50],
                count=len(results),
            )

            return results

    async def record_healing_outcome(
        self,
        pattern_id: str,
        success: bool,
    ) -> None:
        """Record the outcome of applying a healing pattern.

        This updates the success/failure counts for a pattern,
        allowing the system to learn which patterns are reliable.

        Args:
            pattern_id: ID of the pattern to update
            success: Whether the healing was successful
        """
        pool = await self._get_pool()

        try:
            pattern_uuid = uuid.UUID(pattern_id)
        except ValueError:
            self._log.error("Invalid pattern ID", pattern_id=pattern_id)
            return

        async with pool.acquire() as conn:
            if success:
                await conn.execute(
                    """
                    UPDATE test_failure_patterns
                    SET success_count = success_count + 1,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    pattern_uuid,
                )
            else:
                await conn.execute(
                    """
                    UPDATE test_failure_patterns
                    SET failure_count = failure_count + 1,
                        updated_at = NOW()
                    WHERE id = $1
                    """,
                    pattern_uuid,
                )

        self._log.debug(
            "Recorded healing outcome",
            pattern_id=pattern_id,
            success=success,
        )

    async def get_pattern_stats(self) -> dict[str, Any]:
        """Get statistics about stored failure patterns.

        Returns:
            Dictionary with pattern statistics
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_patterns,
                    SUM(success_count) as total_successes,
                    SUM(failure_count) as total_failures,
                    COUNT(DISTINCT error_type) as unique_error_types,
                    AVG(CASE
                        WHEN (success_count + failure_count) > 0
                        THEN success_count::FLOAT / (success_count + failure_count)
                        ELSE NULL
                    END) as avg_success_rate
                FROM test_failure_patterns
                """
            )

            return {
                "total_patterns": row["total_patterns"],
                "total_successes": row["total_successes"] or 0,
                "total_failures": row["total_failures"] or 0,
                "unique_error_types": row["unique_error_types"],
                "avg_success_rate": float(row["avg_success_rate"]) if row["avg_success_rate"] else 0,
            }


# =========================================================================
# Global Instance Management
# =========================================================================

_memory_store: MemoryStore | None = None


def get_memory_store(
    embeddings: Any | None = None,
    database_url: str | None = None,
) -> MemoryStore:
    """Get or create the global memory store instance.

    Args:
        embeddings: Optional embeddings instance for semantic search
        database_url: Optional database URL (defaults to DATABASE_URL env var)

    Returns:
        MemoryStore instance
    """
    global _memory_store

    if _memory_store is None:
        _memory_store = MemoryStore(
            database_url=database_url,
            embeddings=embeddings,
        )

    return _memory_store


def reset_memory_store() -> None:
    """Reset the global memory store instance.

    Useful for testing or when configuration changes.
    """
    global _memory_store
    _memory_store = None


async def init_memory_store(
    embeddings: Any | None = None,
    database_url: str | None = None,
) -> MemoryStore:
    """Initialize and return the memory store.

    This also verifies the database connection.

    Args:
        embeddings: Optional embeddings instance for semantic search
        database_url: Optional database URL

    Returns:
        Initialized MemoryStore instance
    """
    store = get_memory_store(embeddings=embeddings, database_url=database_url)

    # Verify connection by getting the pool
    await store._get_pool()

    logger.info("Memory store initialized successfully")
    return store
