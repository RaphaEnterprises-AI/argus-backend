"""Codebase Retriever - Semantic search over indexed code.

This module provides semantic search capabilities for finding relevant code
in a codebase. It works with the IncrementalIndexer to search through
indexed code chunks.

Search methods:
1. Full-text search using PostgreSQL tsvector
2. Vector semantic search using pgvector embeddings
3. Hybrid search combining both (default)

Usage:
    retriever = CodebaseRetriever(database_url="postgresql://...")

    results = await retriever.search(
        query="login form validation",
        project_id="proj-123",
        limit=5,
    )

    for result in results:
        print(f"{result.file_path}:{result.start_line} - {result.snippet}")
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class SearchSource(Enum):
    """Source of a search result."""
    FULLTEXT = "fulltext"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class CodeSearchResult:
    """A code search result with metadata.

    Attributes:
        chunk_id: Unique identifier for the code chunk
        file_path: Path to the source file
        language: Programming language
        chunk_type: Type of code chunk (function, class, module, etc.)
        name: Name of the code element (function/class name)
        full_name: Fully qualified name
        start_line: Starting line number
        end_line: Ending line number
        snippet: Code snippet (first ~10 lines)
        content: Full chunk content
        score: Relevance score (0-1)
        source: Search method that found this result
        highlights: List of matching terms/phrases
        metadata: Additional metadata
    """
    chunk_id: str
    file_path: str
    language: str
    chunk_type: str
    name: str | None
    full_name: str | None
    start_line: int
    end_line: int
    snippet: str
    content: str
    score: float
    source: SearchSource = SearchSource.HYBRID
    highlights: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "full_name": self.full_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "snippet": self.snippet,
            "content": self.content,
            "score": self.score,
            "source": self.source.value,
            "highlights": self.highlights,
            "metadata": self.metadata,
        }


class CodebaseRetriever:
    """Semantic search retriever for indexed codebases.

    This retriever provides:
    1. Full-text search using PostgreSQL FTS
    2. Vector semantic search using embeddings
    3. Hybrid search combining both approaches

    Example:
        ```python
        retriever = CodebaseRetriever()

        results = await retriever.search(
            query="authentication middleware",
            project_id="my-project",
            limit=5,
            file_types=["py", "ts"],
        )

        for result in results:
            print(f"{result.file_path}:{result.start_line}")
            print(result.snippet)
        ```
    """

    # Maximum snippet length in characters
    MAX_SNIPPET_LENGTH = 500

    # Number of context lines to include in snippet
    SNIPPET_CONTEXT_LINES = 10

    def __init__(
        self,
        database_url: str | None = None,
        embeddings: Any | None = None,
    ):
        """Initialize the codebase retriever.

        Args:
            database_url: PostgreSQL connection URL
            embeddings: LangChain embeddings instance for vector search
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.embeddings = embeddings
        self._pool = None
        self._log = logger.bind(component="codebase_retriever")

    async def _get_pool(self):
        """Get or create the database connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for codebase retriever. Install with: pip install asyncpg"
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

    def _create_snippet(self, content: str, max_lines: int = 10) -> str:
        """Create a code snippet from content.

        Args:
            content: Full code content
            max_lines: Maximum lines to include

        Returns:
            Truncated snippet
        """
        lines = content.split("\n")
        if len(lines) <= max_lines:
            return content

        snippet_lines = lines[:max_lines]
        remaining = len(lines) - max_lines
        snippet = "\n".join(snippet_lines)

        if len(snippet) > self.MAX_SNIPPET_LENGTH:
            snippet = snippet[:self.MAX_SNIPPET_LENGTH] + "..."

        if remaining > 0:
            snippet += f"\n\n// ... {remaining} more lines"

        return snippet

    async def search(
        self,
        query: str,
        project_id: str,
        limit: int = 5,
        file_types: list[str] | None = None,
        chunk_types: list[str] | None = None,
        path_filter: str | None = None,
        use_vector: bool = True,
    ) -> list[CodeSearchResult]:
        """Search the codebase for relevant code.

        Args:
            query: Natural language search query
            project_id: Project ID to search within
            limit: Maximum number of results
            file_types: Filter by file extensions (e.g., ["py", "ts"])
            chunk_types: Filter by chunk types (e.g., ["function", "class"])
            path_filter: Filter by file path pattern (e.g., "src/api/")
            use_vector: Whether to use vector search (requires embeddings)

        Returns:
            List of CodeSearchResult sorted by relevance
        """
        pool = await self._get_pool()

        # Build the query conditions
        conditions = ["project_id = $1"]
        params: list[Any] = [project_id]
        param_idx = 2

        if file_types:
            # Extract extension from file_path
            type_conditions = []
            for ext in file_types:
                type_conditions.append(f"file_path ILIKE $${param_idx}")
                params.append(f"%.{ext}")
                param_idx += 1
            conditions.append(f"({' OR '.join(type_conditions)})")

        if chunk_types:
            type_placeholders = [f"${param_idx + i}" for i in range(len(chunk_types))]
            conditions.append(f"chunk_type IN ({', '.join(type_placeholders)})")
            params.extend(chunk_types)
            param_idx += len(chunk_types)

        if path_filter:
            conditions.append(f"file_path ILIKE ${param_idx}")
            params.append(f"%{path_filter}%")
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Try vector search first if embeddings available
        query_embedding = None
        if use_vector and self.embeddings:
            query_embedding = await self._get_embedding(query)

        results = []

        if query_embedding:
            # Vector search with pgvector
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT
                        chunk_id,
                        file_path,
                        language,
                        chunk_type,
                        name,
                        full_name,
                        start_line,
                        end_line,
                        content,
                        1 - (embedding <=> $${param_idx}::vector) as similarity
                    FROM code_chunks
                    WHERE {where_clause}
                        AND embedding IS NOT NULL
                    ORDER BY embedding <=> $${param_idx}::vector
                    LIMIT ${param_idx + 1}
                    """,
                    *params,
                    embedding_str,
                    limit,
                )

            for row in rows:
                content = row["content"] or ""
                results.append(CodeSearchResult(
                    chunk_id=row["chunk_id"],
                    file_path=row["file_path"],
                    language=row["language"] or "text",
                    chunk_type=row["chunk_type"] or "unknown",
                    name=row["name"],
                    full_name=row["full_name"],
                    start_line=row["start_line"] or 1,
                    end_line=row["end_line"] or 1,
                    snippet=self._create_snippet(content),
                    content=content,
                    score=float(row["similarity"]) if row["similarity"] else 0.0,
                    source=SearchSource.VECTOR,
                ))
        else:
            # Fall back to full-text search
            # Add the search query to params
            params.append(query)
            search_param_idx = param_idx
            param_idx += 1

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT
                        chunk_id,
                        file_path,
                        language,
                        chunk_type,
                        name,
                        full_name,
                        start_line,
                        end_line,
                        content,
                        ts_rank(
                            to_tsvector('english', COALESCE(content, '') || ' ' || COALESCE(name, '')),
                            plainto_tsquery('english', ${search_param_idx})
                        ) as rank
                    FROM code_chunks
                    WHERE {where_clause}
                        AND to_tsvector('english', COALESCE(content, '') || ' ' || COALESCE(name, ''))
                            @@ plainto_tsquery('english', ${search_param_idx})
                    ORDER BY rank DESC
                    LIMIT ${param_idx}
                    """,
                    *params,
                    limit,
                )

            for row in rows:
                content = row["content"] or ""
                results.append(CodeSearchResult(
                    chunk_id=row["chunk_id"],
                    file_path=row["file_path"],
                    language=row["language"] or "text",
                    chunk_type=row["chunk_type"] or "unknown",
                    name=row["name"],
                    full_name=row["full_name"],
                    start_line=row["start_line"] or 1,
                    end_line=row["end_line"] or 1,
                    snippet=self._create_snippet(content),
                    content=content,
                    score=float(row["rank"]) if row["rank"] else 0.0,
                    source=SearchSource.FULLTEXT,
                    highlights=self._extract_highlights(content, query),
                ))

        self._log.info(
            "Codebase search completed",
            query_length=len(query),
            project_id=project_id,
            num_results=len(results),
            used_vector=query_embedding is not None,
        )

        return results

    def _extract_highlights(self, content: str, query: str) -> list[str]:
        """Extract matching terms from content.

        Args:
            content: Code content
            query: Search query

        Returns:
            List of matching terms/phrases
        """
        highlights = []
        query_terms = query.lower().split()
        content_lower = content.lower()

        for term in query_terms:
            if term in content_lower:
                highlights.append(term)

        return highlights[:5]  # Limit to 5 highlights

    async def search_by_name(
        self,
        name: str,
        project_id: str,
        limit: int = 10,
    ) -> list[CodeSearchResult]:
        """Search for code elements by name.

        Args:
            name: Name to search for (function, class, variable)
            project_id: Project ID to search within
            limit: Maximum number of results

        Returns:
            List of matching code elements
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    chunk_id,
                    file_path,
                    language,
                    chunk_type,
                    name,
                    full_name,
                    start_line,
                    end_line,
                    content
                FROM code_chunks
                WHERE project_id = $1
                    AND (name ILIKE $2 OR full_name ILIKE $2)
                ORDER BY
                    CASE WHEN name = $3 THEN 0
                         WHEN name ILIKE $3 THEN 1
                         WHEN full_name ILIKE $2 THEN 2
                         ELSE 3 END,
                    name
                LIMIT $4
                """,
                project_id,
                f"%{name}%",
                name,
                limit,
            )

        results = []
        for row in rows:
            content = row["content"] or ""
            # Calculate a simple score based on match quality
            name_match = row["name"] or ""
            if name_match.lower() == name.lower():
                score = 1.0
            elif name.lower() in name_match.lower():
                score = 0.8
            else:
                score = 0.5

            results.append(CodeSearchResult(
                chunk_id=row["chunk_id"],
                file_path=row["file_path"],
                language=row["language"] or "text",
                chunk_type=row["chunk_type"] or "unknown",
                name=row["name"],
                full_name=row["full_name"],
                start_line=row["start_line"] or 1,
                end_line=row["end_line"] or 1,
                snippet=self._create_snippet(content),
                content=content,
                score=score,
                source=SearchSource.FULLTEXT,
            ))

        return results

    async def get_file_chunks(
        self,
        file_path: str,
        project_id: str,
    ) -> list[CodeSearchResult]:
        """Get all code chunks for a specific file.

        Args:
            file_path: Path to the file
            project_id: Project ID

        Returns:
            List of chunks in the file, ordered by line number
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    chunk_id,
                    file_path,
                    language,
                    chunk_type,
                    name,
                    full_name,
                    start_line,
                    end_line,
                    content
                FROM code_chunks
                WHERE project_id = $1 AND file_path = $2
                ORDER BY start_line
                """,
                project_id,
                file_path,
            )

        results = []
        for row in rows:
            content = row["content"] or ""
            results.append(CodeSearchResult(
                chunk_id=row["chunk_id"],
                file_path=row["file_path"],
                language=row["language"] or "text",
                chunk_type=row["chunk_type"] or "unknown",
                name=row["name"],
                full_name=row["full_name"],
                start_line=row["start_line"] or 1,
                end_line=row["end_line"] or 1,
                snippet=self._create_snippet(content),
                content=content,
                score=1.0,
                source=SearchSource.FULLTEXT,
            ))

        return results

    async def get_stats(self, project_id: str) -> dict[str, Any]:
        """Get statistics about indexed code for a project.

        Args:
            project_id: Project ID

        Returns:
            Dictionary with index statistics
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Total chunks
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM code_chunks WHERE project_id = $1",
                project_id,
            )

            # Chunks by type
            type_counts = await conn.fetch(
                """
                SELECT chunk_type, COUNT(*) as count
                FROM code_chunks
                WHERE project_id = $1
                GROUP BY chunk_type
                ORDER BY count DESC
                """,
                project_id,
            )

            # Chunks by language
            lang_counts = await conn.fetch(
                """
                SELECT language, COUNT(*) as count
                FROM code_chunks
                WHERE project_id = $1
                GROUP BY language
                ORDER BY count DESC
                """,
                project_id,
            )

            # Files indexed
            file_count = await conn.fetchval(
                "SELECT COUNT(DISTINCT file_path) FROM code_chunks WHERE project_id = $1",
                project_id,
            )

            # Chunks with embeddings
            embedded_count = await conn.fetchval(
                "SELECT COUNT(*) FROM code_chunks WHERE project_id = $1 AND embedding IS NOT NULL",
                project_id,
            )

        return {
            "project_id": project_id,
            "total_chunks": total or 0,
            "total_files": file_count or 0,
            "embedded_chunks": embedded_count or 0,
            "by_type": {row["chunk_type"]: row["count"] for row in type_counts},
            "by_language": {row["language"]: row["count"] for row in lang_counts},
        }


# Global instance management
_codebase_retriever: CodebaseRetriever | None = None


def get_codebase_retriever(
    database_url: str | None = None,
    embeddings: Any | None = None,
) -> CodebaseRetriever:
    """Get or create the global codebase retriever instance.

    Args:
        database_url: Database URL (only used on first call)
        embeddings: Embeddings instance (only used on first call)

    Returns:
        CodebaseRetriever instance
    """
    global _codebase_retriever

    if _codebase_retriever is None:
        _codebase_retriever = CodebaseRetriever(
            database_url=database_url,
            embeddings=embeddings,
        )

    return _codebase_retriever


def reset_codebase_retriever() -> None:
    """Reset the global codebase retriever instance."""
    global _codebase_retriever
    _codebase_retriever = None


async def init_codebase_retriever(
    database_url: str | None = None,
    embeddings: Any | None = None,
) -> CodebaseRetriever:
    """Initialize and verify the codebase retriever.

    Args:
        database_url: Database URL
        embeddings: Embeddings instance

    Returns:
        Initialized CodebaseRetriever instance
    """
    retriever = get_codebase_retriever(
        database_url=database_url,
        embeddings=embeddings,
    )

    # Verify connection
    await retriever._get_pool()

    logger.info(
        "Codebase retriever initialized successfully",
        embeddings_available=retriever.embeddings is not None,
    )

    return retriever
