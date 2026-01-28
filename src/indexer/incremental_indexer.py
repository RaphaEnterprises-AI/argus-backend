"""Incremental Indexer - Delta-aware codebase indexing for 10x faster analysis.

The key insight: Most codebase changes are small deltas. Instead of
re-indexing the entire codebase on every commit, we:

1. Track the last indexed commit per project
2. Get the diff between last indexed and current HEAD
3. Only re-parse and re-embed changed files
4. Update the index incrementally

Performance targets:
- Initial full index: 1 file/sec (acceptable for first run)
- Incremental update: 100+ files/sec (only changed files)
- Memory: O(delta) instead of O(codebase)

This is critical for large monorepos where full re-indexing would
take hours and cost $100s in embedding API calls.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src.services.supabase_client import get_supabase_client

from .change_manifest import ChangeManifest, FileChange, IndexUpdate
from .git_integration import GitIntegration, get_git_integration
from .merkle_tree import ChangeSet
from .semantic_chunker import CodeChunk, SemanticChunker, get_chunker
from .tree_sitter_parser import TreeSitterParser, get_parser

logger = structlog.get_logger()


@dataclass
class IndexState:
    """Persisted state of the index for a project.

    Stored in Supabase to track what has been indexed.
    """
    project_id: str
    last_indexed_commit: str | None = None
    last_indexed_at: datetime | None = None
    total_files: int = 0
    total_chunks: int = 0
    index_version: int = 1  # For schema migrations

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "project_id": self.project_id,
            "last_indexed_commit": self.last_indexed_commit,
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "index_version": self.index_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IndexState":
        """Create from dictionary."""
        return cls(
            project_id=data["project_id"],
            last_indexed_commit=data.get("last_indexed_commit"),
            last_indexed_at=datetime.fromisoformat(data["last_indexed_at"]) if data.get("last_indexed_at") else None,
            total_files=data.get("total_files", 0),
            total_chunks=data.get("total_chunks", 0),
            index_version=data.get("index_version", 1),
        )


@dataclass
class ChunkUpdate:
    """Represents a chunk that needs updating in the index."""
    chunk: CodeChunk
    action: str  # "add", "update", "remove"
    file_path: str


class IncrementalIndexer:
    """Delta-aware codebase indexing for efficient updates.

    This is the main orchestrator for incremental indexing. It:
    1. Tracks indexed state per project in Supabase
    2. Uses GitIntegration to get changes between commits
    3. Uses TreeSitterParser and SemanticChunker for parsing
    4. Coordinates with LocalEmbedder for vector generation

    Usage:
        indexer = IncrementalIndexer("/path/to/repo")

        # Get what commit was last indexed
        last_commit = await indexer.get_last_indexed_commit("project-123")

        # Analyze changes since last index
        changes = await indexer.analyze_changes(
            repo_url="https://github.com/org/repo",
            from_commit=last_commit,
            to_commit="HEAD"
        )

        # Update the index with only changed files
        result = await indexer.update_index("project-123", changes)

        # Save the new indexed state
        await indexer.save_indexed_commit("project-123", "abc123")
    """

    # Table name for index state in Supabase
    INDEX_STATE_TABLE = "codebase_index_state"

    # Maximum files to process in parallel
    MAX_PARALLEL_FILES = 20

    # Chunk batch size for embedding
    CHUNK_BATCH_SIZE = 50

    def __init__(
        self,
        repo_path: str | None = None,
        repo_url: str | None = None,
        parser: TreeSitterParser | None = None,
        chunker: SemanticChunker | None = None,
        git: GitIntegration | None = None,
    ):
        """Initialize the incremental indexer.

        Args:
            repo_path: Path to local repository
            repo_url: URL for remote repository
            parser: Optional TreeSitterParser instance
            chunker: Optional SemanticChunker instance
            git: Optional GitIntegration instance
        """
        self.repo_path = Path(repo_path).resolve() if repo_path else Path.cwd()
        self.repo_url = repo_url

        # Use provided instances or get global ones
        self._parser = parser or get_parser()
        self._chunker = chunker or get_chunker()
        self._git = git or GitIntegration(
            repo_path=str(self.repo_path),
            repo_url=repo_url,
        )

        # Supabase client for state persistence
        self._supabase = get_supabase_client()

        # Semaphore for parallel processing
        self._semaphore = asyncio.Semaphore(self.MAX_PARALLEL_FILES)

        # Local cache for index state
        self._state_cache: dict[str, IndexState] = {}

    async def get_last_indexed_commit(self, project_id: str) -> str | None:
        """Get the last indexed commit SHA for a project.

        Args:
            project_id: Project identifier

        Returns:
            Commit SHA or None if never indexed
        """
        # Check cache first
        if project_id in self._state_cache:
            return self._state_cache[project_id].last_indexed_commit

        # Query Supabase
        if not self._supabase.is_configured:
            logger.warning("Supabase not configured, cannot get index state")
            return None

        result = await self._supabase.select(
            self.INDEX_STATE_TABLE,
            columns="*",
            filters={"project_id": f"eq.{project_id}"},
        )

        if result["error"]:
            logger.error("Failed to get index state", error=result["error"])
            return None

        data = result["data"]
        if not data or len(data) == 0:
            return None

        # Cache it
        state = IndexState.from_dict(data[0])
        self._state_cache[project_id] = state

        return state.last_indexed_commit

    async def analyze_changes(
        self,
        repo_url: str,
        from_commit: str,
        to_commit: str,
    ) -> ChangeManifest:
        """Analyze changes between two commits.

        This is the entry point for understanding what needs updating.

        Args:
            repo_url: Repository URL
            from_commit: Starting commit SHA (base)
            to_commit: Ending commit SHA (head)

        Returns:
            ChangeManifest with all file changes
        """
        return await self._git.analyze_changes(repo_url, from_commit, to_commit)

    async def update_index(
        self,
        project_id: str,
        changes: ChangeManifest,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IndexUpdate:
        """Update the index based on a change manifest.

        This is the core incremental indexing logic:
        1. For deleted files: Remove chunks from index
        2. For added/modified files: Re-parse and re-chunk
        3. Update embeddings for new/changed chunks
        4. Return statistics about what was updated

        Args:
            project_id: Project identifier
            changes: ChangeManifest from analyze_changes
            progress_callback: Optional callback(file_path, current, total)

        Returns:
            IndexUpdate with statistics
        """
        start_time = time.perf_counter()

        result = IndexUpdate(
            project_id=project_id,
            from_commit=changes.from_commit,
            to_commit=changes.to_commit,
        )

        if not changes.has_changes:
            logger.info("No changes to index", project_id=project_id)
            return result

        # Filter to only source files worth indexing
        source_changes = changes.source_files
        logger.info(
            "Processing changes",
            project_id=project_id,
            total_files=len(source_changes),
            added=len(changes.added_files),
            modified=len(changes.modified_files),
            deleted=len(changes.deleted_files),
            renamed=len(changes.renamed_files),
        )

        # 1. Handle deleted files (and old paths from renames)
        files_to_remove = changes.files_to_remove
        if files_to_remove:
            removed_chunks = await self._remove_files_from_index(
                project_id, files_to_remove
            )
            result.files_removed = len(files_to_remove)
            result.chunks_removed = removed_chunks

        # 2. Handle added/modified files
        files_to_index = [
            f for f in source_changes
            if f.change_type in ("added", "modified", "renamed")
        ]

        if files_to_index:
            indexed_count, chunk_count, errors = await self._index_files(
                project_id,
                files_to_index,
                progress_callback,
            )
            result.files_indexed = indexed_count
            result.chunks_added = chunk_count
            result.errors = errors

        result.duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Index update complete",
            project_id=project_id,
            files_indexed=result.files_indexed,
            files_removed=result.files_removed,
            chunks_added=result.chunks_added,
            chunks_removed=result.chunks_removed,
            duration_ms=result.duration_ms,
        )

        return result

    async def _index_files(
        self,
        project_id: str,
        files: list[FileChange],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[int, int, list[str]]:
        """Index a list of files in parallel.

        Args:
            project_id: Project identifier
            files: Files to index
            progress_callback: Optional progress callback

        Returns:
            Tuple of (files_indexed, chunks_created, errors)
        """
        indexed = 0
        total_chunks = 0
        errors = []
        total = len(files)

        # Process files in parallel with semaphore limiting
        async def process_file(file_change: FileChange, index: int) -> tuple[int, str | None]:
            async with self._semaphore:
                try:
                    file_path = self.repo_path / file_change.path
                    if not file_path.exists():
                        return 0, f"File not found: {file_change.path}"

                    # Parse the file
                    parsed = self._parser.parse_file(str(file_path))

                    if parsed.has_errors:
                        logger.warning(
                            "Parse errors",
                            file=file_change.path,
                            errors=parsed.errors,
                        )

                    # Chunk it
                    chunks = self._chunker.chunk(parsed)

                    # Store chunks (implementation depends on vector store)
                    await self._store_chunks(project_id, file_change.path, chunks)

                    if progress_callback:
                        progress_callback(file_change.path, index + 1, total)

                    return len(chunks), None

                except Exception as e:
                    logger.error(
                        "Failed to index file",
                        file=file_change.path,
                        error=str(e),
                    )
                    return 0, f"{file_change.path}: {str(e)}"

        # Run all files in parallel
        tasks = [
            process_file(f, i)
            for i, f in enumerate(files)
        ]
        results = await asyncio.gather(*tasks)

        for chunk_count, error in results:
            if error:
                errors.append(error)
            else:
                indexed += 1
                total_chunks += chunk_count

        return indexed, total_chunks, errors

    async def _remove_files_from_index(
        self,
        project_id: str,
        file_paths: list[str],
    ) -> int:
        """Remove files from the index.

        Args:
            project_id: Project identifier
            file_paths: Paths to remove

        Returns:
            Number of chunks removed
        """
        total_removed = 0

        for path in file_paths:
            try:
                removed = await self._delete_chunks_for_file(project_id, path)
                total_removed += removed
                logger.debug("Removed chunks", file=path, count=removed)
            except Exception as e:
                logger.error("Failed to remove file chunks", file=path, error=str(e))

        return total_removed

    async def _store_chunks(
        self,
        project_id: str,
        file_path: str,
        chunks: list[CodeChunk],
    ) -> None:
        """Store chunks in the vector database.

        This is where embeddings would be generated and stored.
        The actual implementation depends on your vector store (pgvector, Cloudflare Vectorize, etc.)

        Args:
            project_id: Project identifier
            file_path: Source file path
            chunks: Chunks to store
        """
        if not self._supabase.is_configured:
            logger.debug("Supabase not configured, skipping chunk storage")
            return

        # For now, we store chunk metadata in Supabase
        # Embeddings would be generated by LocalEmbedder and stored in pgvector
        for chunk in chunks:
            chunk_data = {
                "project_id": project_id,
                "file_path": file_path,
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type.value,
                "name": chunk.name,
                "full_name": chunk.full_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content_hash": chunk.content_hash,
                "language": chunk.language,
                "created_at": datetime.now(UTC).isoformat(),
            }

            # This would be an upsert to handle updates
            await self._supabase.request(
                "/code_chunks",
                method="POST",
                body=chunk_data,
                headers={"Prefer": "resolution=merge-duplicates"},
            )

    async def _delete_chunks_for_file(
        self,
        project_id: str,
        file_path: str,
    ) -> int:
        """Delete all chunks for a file.

        Args:
            project_id: Project identifier
            file_path: File path

        Returns:
            Number of chunks deleted
        """
        if not self._supabase.is_configured:
            return 0

        result = await self._supabase.request(
            f"/code_chunks?project_id=eq.{project_id}&file_path=eq.{file_path}",
            method="DELETE",
            headers={"Prefer": "return=representation"},
        )

        if result["error"]:
            return 0

        return len(result["data"]) if result["data"] else 0

    async def save_indexed_commit(
        self,
        project_id: str,
        commit_sha: str,
        stats: dict | None = None,
    ) -> bool:
        """Save the indexed commit for a project.

        Called after successful index update to persist state.

        Args:
            project_id: Project identifier
            commit_sha: The commit SHA that was indexed
            stats: Optional statistics to store

        Returns:
            True if saved successfully
        """
        if not self._supabase.is_configured:
            logger.warning("Supabase not configured, cannot save index state")
            # Update local cache at least
            self._state_cache[project_id] = IndexState(
                project_id=project_id,
                last_indexed_commit=commit_sha,
                last_indexed_at=datetime.now(UTC),
            )
            return True

        state = IndexState(
            project_id=project_id,
            last_indexed_commit=commit_sha,
            last_indexed_at=datetime.now(UTC),
            total_files=stats.get("total_files", 0) if stats else 0,
            total_chunks=stats.get("total_chunks", 0) if stats else 0,
        )

        result = await self._supabase.request(
            f"/{self.INDEX_STATE_TABLE}",
            method="POST",
            body=state.to_dict(),
            headers={"Prefer": "resolution=merge-duplicates"},
        )

        if result["error"]:
            logger.error("Failed to save index state", error=result["error"])
            return False

        # Update cache
        self._state_cache[project_id] = state
        return True

    async def full_index(
        self,
        project_id: str,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IndexUpdate:
        """Perform a full index of the repository.

        Used for initial indexing when there's no previous state.

        Args:
            project_id: Project identifier
            progress_callback: Optional progress callback

        Returns:
            IndexUpdate with statistics
        """
        start_time = time.perf_counter()

        # Get current commit
        current_commit = await self._git.get_current_commit()
        if not current_commit:
            logger.error("Could not determine current commit")
            return IndexUpdate(
                project_id=project_id,
                from_commit="",
                to_commit="",
                errors=["Could not determine current commit"],
            )

        # List all files
        all_files = await self._git.list_files_at_commit(current_commit)

        # Filter to source files
        source_files = [
            f for f in all_files
            if any(f.endswith(ext) for ext in self._git.SOURCE_EXTENSIONS)
        ]

        logger.info(
            "Starting full index",
            project_id=project_id,
            total_files=len(source_files),
            commit=current_commit[:8],
        )

        # Create file changes for all files (treat as "added")
        file_changes = [
            FileChange(path=f, change_type="added")
            for f in source_files
        ]

        indexed, chunks, errors = await self._index_files(
            project_id,
            file_changes,
            progress_callback,
        )

        # Save state
        await self.save_indexed_commit(
            project_id,
            current_commit,
            {"total_files": indexed, "total_chunks": chunks},
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Full index complete",
            project_id=project_id,
            files_indexed=indexed,
            chunks_created=chunks,
            duration_ms=duration_ms,
        )

        return IndexUpdate(
            project_id=project_id,
            from_commit="",
            to_commit=current_commit,
            files_indexed=indexed,
            chunks_added=chunks,
            duration_ms=duration_ms,
            errors=errors,
        )

    async def smart_index(
        self,
        project_id: str,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> IndexUpdate:
        """Intelligently decide between full or incremental index.

        This is the recommended entry point. It will:
        1. Check if there's a previous index state
        2. If no previous state, do full index
        3. If previous state exists, do incremental update

        Args:
            project_id: Project identifier
            progress_callback: Optional progress callback

        Returns:
            IndexUpdate with statistics
        """
        # Get last indexed commit
        last_commit = await self.get_last_indexed_commit(project_id)

        if last_commit is None:
            logger.info("No previous index, performing full index", project_id=project_id)
            return await self.full_index(project_id, progress_callback)

        # Get current commit
        current_commit = await self._git.get_current_commit()
        if not current_commit:
            return IndexUpdate(
                project_id=project_id,
                from_commit=last_commit,
                to_commit="",
                errors=["Could not determine current commit"],
            )

        if current_commit == last_commit:
            logger.info("Index is up to date", project_id=project_id, commit=current_commit[:8])
            return IndexUpdate(
                project_id=project_id,
                from_commit=last_commit,
                to_commit=current_commit,
            )

        # Analyze changes
        changes = await self.analyze_changes(
            self.repo_url or "",
            last_commit,
            current_commit,
        )

        logger.info(
            "Performing incremental update",
            project_id=project_id,
            from_commit=last_commit[:8],
            to_commit=current_commit[:8],
            files_changed=changes.total_changes,
        )

        # Update index
        result = await self.update_index(project_id, changes, progress_callback)

        # Save new state
        if result.success:
            await self.save_indexed_commit(
                project_id,
                current_commit,
                {"total_files": result.files_indexed, "total_chunks": result.chunks_added},
            )

        return result

    def get_stats(self, project_id: str) -> dict:
        """Get statistics about the index for a project.

        Args:
            project_id: Project identifier

        Returns:
            Dictionary with index statistics
        """
        state = self._state_cache.get(project_id)
        if not state:
            return {"status": "not_indexed", "project_id": project_id}

        return {
            "status": "indexed",
            "project_id": project_id,
            "last_indexed_commit": state.last_indexed_commit,
            "last_indexed_at": state.last_indexed_at.isoformat() if state.last_indexed_at else None,
            "total_files": state.total_files,
            "total_chunks": state.total_chunks,
            "index_version": state.index_version,
        }


# Global instance (lazy initialized)
_incremental_indexer: IncrementalIndexer | None = None


def get_incremental_indexer(
    repo_path: str = ".",
    repo_url: str | None = None,
) -> IncrementalIndexer:
    """Get or create IncrementalIndexer instance.

    Args:
        repo_path: Path to repository
        repo_url: Optional repository URL

    Returns:
        IncrementalIndexer instance
    """
    global _incremental_indexer

    if _incremental_indexer is None:
        _incremental_indexer = IncrementalIndexer(
            repo_path=repo_path,
            repo_url=repo_url,
        )

    return _incremental_indexer
