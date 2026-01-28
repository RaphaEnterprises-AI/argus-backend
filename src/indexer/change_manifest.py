"""Change Manifest - Data structures for tracking codebase changes.

This module provides data classes for representing:
- Individual file changes (added, modified, deleted, renamed)
- Change manifests summarizing diffs between commits
- Index update results with statistics

Used by IncrementalIndexer to efficiently track what needs re-indexing.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Literal


class ChangeType(str, Enum):
    """Types of file changes."""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    TYPE_CHANGED = "type_changed"


@dataclass
class FileChange:
    """Represents a single file change between commits.

    Attributes:
        path: Current file path
        change_type: Type of change (added, modified, deleted, renamed)
        old_path: Previous path for renames, None otherwise
        additions: Number of lines added
        deletions: Number of lines deleted
        binary: Whether file is binary
        content_hash: SHA of the file content (for modified/added files)
        old_content_hash: SHA of previous content (for modified files)
    """
    path: str
    change_type: Literal["added", "modified", "deleted", "renamed"]
    old_path: str | None = None  # For renames
    additions: int = 0
    deletions: int = 0
    binary: bool = False
    content_hash: str | None = None
    old_content_hash: str | None = None

    @property
    def is_source_file(self) -> bool:
        """Check if this is a source code file worth indexing."""
        source_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
            ".kt", ".scala", ".sql", ".sh", ".yaml", ".yml", ".json",
            ".toml", ".md", ".vue", ".svelte",
        }
        return any(self.path.endswith(ext) for ext in source_extensions)

    @property
    def affected_paths(self) -> list[str]:
        """Get all paths affected by this change (includes old_path for renames)."""
        paths = [self.path]
        if self.old_path:
            paths.append(self.old_path)
        return paths

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "change_type": self.change_type,
            "old_path": self.old_path,
            "additions": self.additions,
            "deletions": self.deletions,
            "binary": self.binary,
            "content_hash": self.content_hash,
            "old_content_hash": self.old_content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileChange":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            change_type=data["change_type"],
            old_path=data.get("old_path"),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            binary=data.get("binary", False),
            content_hash=data.get("content_hash"),
            old_content_hash=data.get("old_content_hash"),
        )


@dataclass
class ChangeManifest:
    """Summary of changes between two commits.

    Attributes:
        from_commit: Starting commit SHA (base)
        to_commit: Ending commit SHA (head)
        files: List of file changes
        total_additions: Total lines added across all files
        total_deletions: Total lines deleted across all files
        timestamp: When the manifest was created
    """
    from_commit: str
    to_commit: str
    files: list[FileChange]
    total_additions: int = 0
    total_deletions: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Calculate totals if not provided."""
        if self.total_additions == 0 and self.total_deletions == 0:
            self.total_additions = sum(f.additions for f in self.files)
            self.total_deletions = sum(f.deletions for f in self.files)

    @property
    def total_changes(self) -> int:
        """Total number of file changes."""
        return len(self.files)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return len(self.files) > 0

    @property
    def added_files(self) -> list[FileChange]:
        """Get all added files."""
        return [f for f in self.files if f.change_type == "added"]

    @property
    def modified_files(self) -> list[FileChange]:
        """Get all modified files."""
        return [f for f in self.files if f.change_type == "modified"]

    @property
    def deleted_files(self) -> list[FileChange]:
        """Get all deleted files."""
        return [f for f in self.files if f.change_type == "deleted"]

    @property
    def renamed_files(self) -> list[FileChange]:
        """Get all renamed files."""
        return [f for f in self.files if f.change_type == "renamed"]

    @property
    def source_files(self) -> list[FileChange]:
        """Get only source code files (worth indexing)."""
        return [f for f in self.files if f.is_source_file]

    @property
    def files_to_index(self) -> list[str]:
        """Get paths that need indexing (added + modified + new path of renamed)."""
        paths = []
        for f in self.files:
            if f.change_type in ("added", "modified", "renamed"):
                paths.append(f.path)
        return paths

    @property
    def files_to_remove(self) -> list[str]:
        """Get paths that need removal from index (deleted + old path of renamed)."""
        paths = []
        for f in self.files:
            if f.change_type == "deleted":
                paths.append(f.path)
            elif f.change_type == "renamed" and f.old_path:
                paths.append(f.old_path)
        return paths

    def filter_by_extensions(self, extensions: set[str]) -> "ChangeManifest":
        """Create filtered manifest with only specified extensions.

        Args:
            extensions: Set of extensions to include (e.g., {".py", ".ts"})

        Returns:
            New ChangeManifest with filtered files
        """
        filtered = [
            f for f in self.files
            if any(f.path.endswith(ext) for ext in extensions)
        ]
        return ChangeManifest(
            from_commit=self.from_commit,
            to_commit=self.to_commit,
            files=filtered,
            timestamp=self.timestamp,
        )

    def filter_by_paths(self, path_prefixes: list[str]) -> "ChangeManifest":
        """Create filtered manifest with only files matching path prefixes.

        Args:
            path_prefixes: List of path prefixes to include (e.g., ["src/", "lib/"])

        Returns:
            New ChangeManifest with filtered files
        """
        filtered = [
            f for f in self.files
            if any(f.path.startswith(prefix) for prefix in path_prefixes)
        ]
        return ChangeManifest(
            from_commit=self.from_commit,
            to_commit=self.to_commit,
            files=filtered,
            timestamp=self.timestamp,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "from_commit": self.from_commit,
            "to_commit": self.to_commit,
            "files": [f.to_dict() for f in self.files],
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChangeManifest":
        """Create from dictionary."""
        return cls(
            from_commit=data["from_commit"],
            to_commit=data["to_commit"],
            files=[FileChange.from_dict(f) for f in data["files"]],
            total_additions=data.get("total_additions", 0),
            total_deletions=data.get("total_deletions", 0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(UTC),
        )


@dataclass
class IndexUpdate:
    """Result of an incremental index update.

    Attributes:
        project_id: Project identifier
        from_commit: Starting commit SHA
        to_commit: Ending commit SHA
        files_indexed: Number of files re-indexed
        files_removed: Number of files removed from index
        chunks_added: Number of new chunks created
        chunks_removed: Number of chunks removed
        chunks_updated: Number of chunks updated
        duration_ms: Time taken to perform update
        errors: List of errors encountered
    """
    project_id: str
    from_commit: str
    to_commit: str
    files_indexed: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_updated: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_chunks_affected(self) -> int:
        """Total number of chunks affected by this update."""
        return self.chunks_added + self.chunks_removed + self.chunks_updated

    @property
    def success(self) -> bool:
        """Check if update was successful (no errors)."""
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "from_commit": self.from_commit,
            "to_commit": self.to_commit,
            "files_indexed": self.files_indexed,
            "files_removed": self.files_removed,
            "chunks_added": self.chunks_added,
            "chunks_removed": self.chunks_removed,
            "chunks_updated": self.chunks_updated,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }


@dataclass
class BlameInfo:
    """Git blame information for a specific line.

    Attributes:
        commit_sha: SHA of the commit that last modified this line
        author: Author name
        author_email: Author email
        timestamp: When the line was last modified
        line_number: Line number in the file
        commit_message: Short commit message
    """
    commit_sha: str
    author: str
    author_email: str
    timestamp: datetime
    line_number: int
    commit_message: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "commit_sha": self.commit_sha,
            "author": self.author,
            "author_email": self.author_email,
            "timestamp": self.timestamp.isoformat(),
            "line_number": self.line_number,
            "commit_message": self.commit_message,
        }
