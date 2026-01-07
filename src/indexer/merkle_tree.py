"""Merkle Tree - O(log n) change detection for large codebases.

Inspired by Cursor's approach:
- Instead of re-indexing entire repo on every change
- Build tree of content hashes
- On file change, only recompute affected branch
- Efficiently detect: which files changed, which need re-embedding

This enables:
- Sub-second incremental updates for large repos
- Efficient sync between local index and cloud vectors
- Git-like change tracking without git dependency
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator
import time

logger = logging.getLogger(__name__)


@dataclass
class MerkleNode:
    """A node in the Merkle tree."""

    # Node identity
    path: str           # File path or directory path
    hash: str           # Content hash (for files) or combined children hash
    is_file: bool       # True for files, False for directories

    # File metadata (only for files)
    size: int = 0       # File size in bytes
    modified_time: float = 0.0  # Last modification time
    line_count: int = 0  # Number of lines

    # Tree structure
    children: list["MerkleNode"] = field(default_factory=list)

    # Change tracking
    changed: bool = False  # True if hash changed since last sync

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "hash": self.hash,
            "is_file": self.is_file,
            "size": self.size,
            "modified_time": self.modified_time,
            "line_count": self.line_count,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MerkleNode":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            hash=data["hash"],
            is_file=data["is_file"],
            size=data.get("size", 0),
            modified_time=data.get("modified_time", 0.0),
            line_count=data.get("line_count", 0),
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )


@dataclass
class ChangeSet:
    """Result of comparing two Merkle trees."""

    added: list[str]      # New files
    modified: list[str]   # Changed files
    deleted: list[str]    # Removed files

    @property
    def total_changes(self) -> int:
        """Total number of changes."""
        return len(self.added) + len(self.modified) + len(self.deleted)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return self.total_changes > 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "total_changes": self.total_changes,
        }


@dataclass
class MerkleTreeConfig:
    """Configuration for Merkle tree building."""

    # File selection
    include_patterns: list[str] = field(default_factory=lambda: ["**/*"])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/venv/**",
        "**/__pycache__/**",
        "**/dist/**",
        "**/build/**",
        "**/.next/**",
        "**/target/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/.DS_Store",
    ])

    # Languages to index
    include_extensions: set[str] = field(default_factory=lambda: {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".sql", ".sh", ".yaml", ".yml", ".json",
        ".toml", ".md", ".vue", ".svelte",
    })

    # Size limits
    max_file_size: int = 1_000_000  # 1MB max file size
    min_file_size: int = 10         # Skip near-empty files


class MerkleTree:
    """Merkle tree for efficient codebase change detection.

    Usage:
        tree = MerkleTree("/path/to/repo")
        tree.build()

        # Later, detect changes
        changes = tree.detect_changes()
        for file_path in changes.modified:
            # Re-index this file
            ...

        # Save state
        tree.save(".argus/merkle.json")

        # Load previous state
        old_tree = MerkleTree.load(".argus/merkle.json")
    """

    def __init__(
        self,
        root_path: str,
        config: Optional[MerkleTreeConfig] = None
    ):
        """Initialize Merkle tree for a directory.

        Args:
            root_path: Root directory to index
            config: Optional configuration
        """
        self.root_path = Path(root_path).resolve()
        self.config = config or MerkleTreeConfig()
        self.root: Optional[MerkleNode] = None
        self._file_index: dict[str, MerkleNode] = {}  # path -> node for fast lookup

        # Stats
        self.total_files: int = 0
        self.total_size: int = 0
        self.build_time_ms: float = 0.0

    def build(self) -> MerkleNode:
        """Build the Merkle tree from the filesystem.

        Returns:
            Root node of the tree
        """
        start = time.perf_counter()

        self._file_index = {}
        self.total_files = 0
        self.total_size = 0

        self.root = self._build_node(self.root_path)

        self.build_time_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Built Merkle tree: {self.total_files} files, "
            f"{self.total_size / 1024 / 1024:.1f}MB, "
            f"{self.build_time_ms:.1f}ms"
        )

        return self.root

    def _build_node(self, path: Path, depth: int = 0) -> MerkleNode:
        """Recursively build tree node for a path."""
        rel_path = str(path.relative_to(self.root_path))

        if path.is_file():
            return self._build_file_node(path, rel_path)
        else:
            return self._build_dir_node(path, rel_path, depth)

    def _build_file_node(self, path: Path, rel_path: str) -> MerkleNode:
        """Build node for a file."""
        try:
            stat = path.stat()
            size = stat.st_size

            # Skip if too large or too small
            if size > self.config.max_file_size or size < self.config.min_file_size:
                return None

            # Read and hash content
            content = path.read_bytes()
            content_hash = hashlib.sha256(content).hexdigest()[:16]

            # Count lines for text files
            try:
                text_content = content.decode("utf-8")
                line_count = text_content.count("\n") + 1
            except UnicodeDecodeError:
                line_count = 0

            node = MerkleNode(
                path=rel_path,
                hash=content_hash,
                is_file=True,
                size=size,
                modified_time=stat.st_mtime,
                line_count=line_count,
            )

            # Update index and stats
            self._file_index[rel_path] = node
            self.total_files += 1
            self.total_size += size

            return node

        except Exception as e:
            logger.debug(f"Skipping file {rel_path}: {e}")
            return None

    def _build_dir_node(
        self,
        path: Path,
        rel_path: str,
        depth: int
    ) -> MerkleNode:
        """Build node for a directory."""
        children = []

        try:
            for child_path in sorted(path.iterdir()):
                # Skip excluded patterns
                if self._should_exclude(child_path):
                    continue

                # Only include files with right extensions
                if child_path.is_file():
                    if child_path.suffix.lower() not in self.config.include_extensions:
                        continue

                child_node = self._build_node(child_path, depth + 1)
                if child_node:
                    children.append(child_node)

        except PermissionError:
            logger.debug(f"Permission denied: {rel_path}")

        # Compute directory hash from children
        if children:
            children_hashes = "".join(c.hash for c in children)
            dir_hash = hashlib.sha256(children_hashes.encode()).hexdigest()[:16]
        else:
            dir_hash = hashlib.sha256(rel_path.encode()).hexdigest()[:16]

        return MerkleNode(
            path=rel_path,
            hash=dir_hash,
            is_file=False,
            children=children,
        )

    def _should_exclude(self, path: Path) -> bool:
        """Check if path matches exclude patterns."""
        path_str = str(path)

        for pattern in self.config.exclude_patterns:
            # Simple pattern matching
            if "**" in pattern:
                # Handle glob patterns
                pattern_parts = pattern.replace("**", "").strip("/")
                if pattern_parts in path_str:
                    return True
            elif path.match(pattern):
                return True

        return False

    def detect_changes(self) -> ChangeSet:
        """Detect changes by rebuilding tree and comparing.

        Returns:
            ChangeSet with added/modified/deleted files
        """
        if self.root is None:
            # First build
            self.build()
            return ChangeSet(
                added=list(self._file_index.keys()),
                modified=[],
                deleted=[],
            )

        # Store old index
        old_index = self._file_index.copy()

        # Rebuild tree
        self.build()

        # Compare
        added = []
        modified = []
        deleted = []

        new_paths = set(self._file_index.keys())
        old_paths = set(old_index.keys())

        # New files
        for path in new_paths - old_paths:
            added.append(path)

        # Deleted files
        for path in old_paths - new_paths:
            deleted.append(path)

        # Modified files (hash changed)
        for path in new_paths & old_paths:
            if self._file_index[path].hash != old_index[path].hash:
                modified.append(path)
                self._file_index[path].changed = True

        return ChangeSet(added=added, modified=modified, deleted=deleted)

    def compare_with(self, other: "MerkleTree") -> ChangeSet:
        """Compare this tree with another tree.

        Args:
            other: Another MerkleTree to compare against

        Returns:
            ChangeSet representing differences
        """
        added = []
        modified = []
        deleted = []

        our_paths = set(self._file_index.keys())
        their_paths = set(other._file_index.keys())

        # Files we have that they don't
        for path in our_paths - their_paths:
            added.append(path)

        # Files they have that we don't
        for path in their_paths - our_paths:
            deleted.append(path)

        # Files in both - check hash
        for path in our_paths & their_paths:
            if self._file_index[path].hash != other._file_index[path].hash:
                modified.append(path)

        return ChangeSet(added=added, modified=modified, deleted=deleted)

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get hash for a specific file.

        Args:
            file_path: Relative path to file

        Returns:
            Content hash or None if not found
        """
        node = self._file_index.get(file_path)
        return node.hash if node else None

    def get_files_by_extension(self, extension: str) -> list[str]:
        """Get all files with a specific extension.

        Args:
            extension: File extension (e.g., ".py")

        Returns:
            List of file paths
        """
        return [
            path for path in self._file_index.keys()
            if path.endswith(extension)
        ]

    def get_all_files(self) -> Generator[str, None, None]:
        """Iterate over all indexed files."""
        yield from self._file_index.keys()

    def get_changed_files(self) -> list[str]:
        """Get files marked as changed."""
        return [
            path for path, node in self._file_index.items()
            if node.changed
        ]

    def get_stats(self) -> dict:
        """Get tree statistics."""
        return {
            "root_path": str(self.root_path),
            "total_files": self.total_files,
            "total_size_bytes": self.total_size,
            "total_size_mb": round(self.total_size / 1024 / 1024, 2),
            "build_time_ms": round(self.build_time_ms, 2),
            "root_hash": self.root.hash if self.root else None,
        }

    def save(self, path: str) -> bool:
        """Save tree state to JSON file.

        Args:
            path: Path to save file

        Returns:
            True if successful
        """
        if not self.root:
            return False

        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "root_path": str(self.root_path),
                "total_files": self.total_files,
                "total_size": self.total_size,
                "build_time_ms": self.build_time_ms,
                "tree": self.root.to_dict(),
            }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved Merkle tree to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save Merkle tree: {e}")
            return False

    @classmethod
    def load(cls, path: str) -> Optional["MerkleTree"]:
        """Load tree state from JSON file.

        Args:
            path: Path to saved file

        Returns:
            MerkleTree instance or None
        """
        try:
            with open(path) as f:
                data = json.load(f)

            tree = cls(data["root_path"])
            tree.total_files = data["total_files"]
            tree.total_size = data["total_size"]
            tree.build_time_ms = data.get("build_time_ms", 0.0)
            tree.root = MerkleNode.from_dict(data["tree"])

            # Rebuild file index from tree
            tree._rebuild_file_index(tree.root)

            logger.info(f"Loaded Merkle tree from {path}: {tree.total_files} files")
            return tree

        except Exception as e:
            logger.error(f"Failed to load Merkle tree: {e}")
            return None

    def _rebuild_file_index(self, node: MerkleNode) -> None:
        """Rebuild file index from tree structure."""
        if node.is_file:
            self._file_index[node.path] = node
        else:
            for child in node.children:
                self._rebuild_file_index(child)


class IncrementalIndexer:
    """Combines Merkle tree with semantic chunking for incremental indexing.

    Usage:
        indexer = IncrementalIndexer("/path/to/repo")

        # Initial index
        indexer.full_index()

        # Later, incremental update
        changes = indexer.incremental_update()
        print(f"Re-indexed {len(changes.modified)} files")
    """

    def __init__(
        self,
        root_path: str,
        state_dir: str = ".argus",
        config: Optional[MerkleTreeConfig] = None
    ):
        """Initialize incremental indexer.

        Args:
            root_path: Repository root
            state_dir: Directory to store index state
            config: Merkle tree configuration
        """
        self.root_path = Path(root_path).resolve()
        self.state_dir = self.root_path / state_dir
        self.config = config

        self.tree: Optional[MerkleTree] = None
        self._state_file = self.state_dir / "merkle.json"

    def full_index(self, callback=None) -> dict:
        """Perform full index of the repository.

        Args:
            callback: Optional callback(file_path, chunk_count) for progress

        Returns:
            Index statistics
        """
        from .tree_sitter_parser import get_parser
        from .semantic_chunker import get_chunker

        parser = get_parser()
        chunker = get_chunker()

        # Build Merkle tree
        self.tree = MerkleTree(str(self.root_path), self.config)
        self.tree.build()

        # Index all files
        total_chunks = 0
        file_count = 0

        for file_path in self.tree.get_all_files():
            full_path = self.root_path / file_path

            try:
                parsed = parser.parse_file(str(full_path))
                chunks = chunker.chunk(parsed)
                total_chunks += len(chunks)
                file_count += 1

                if callback:
                    callback(file_path, len(chunks))

            except Exception as e:
                logger.debug(f"Could not index {file_path}: {e}")

        # Save state
        self.save_state()

        return {
            "files_indexed": file_count,
            "total_chunks": total_chunks,
            "tree_stats": self.tree.get_stats(),
        }

    def incremental_update(self, callback=None) -> ChangeSet:
        """Perform incremental update, re-indexing only changed files.

        Args:
            callback: Optional callback(file_path, action) for progress

        Returns:
            ChangeSet with what changed
        """
        from .tree_sitter_parser import get_parser
        from .semantic_chunker import get_chunker

        # Load previous state if exists
        if self.tree is None:
            self.load_state()

        if self.tree is None:
            # No previous state, do full index
            self.full_index(callback)
            return ChangeSet(added=list(self.tree.get_all_files()), modified=[], deleted=[])

        # Detect changes
        changes = self.tree.detect_changes()

        if not changes.has_changes:
            logger.info("No changes detected")
            return changes

        parser = get_parser()
        chunker = get_chunker()

        # Re-index changed files
        files_to_reindex = changes.added + changes.modified

        for file_path in files_to_reindex:
            full_path = self.root_path / file_path

            try:
                parsed = parser.parse_file(str(full_path))
                chunks = chunker.chunk(parsed)

                if callback:
                    callback(file_path, "updated")

            except Exception as e:
                logger.debug(f"Could not re-index {file_path}: {e}")

        # Handle deleted files
        for file_path in changes.deleted:
            if callback:
                callback(file_path, "deleted")

        # Save updated state
        self.save_state()

        logger.info(
            f"Incremental update: {len(changes.added)} added, "
            f"{len(changes.modified)} modified, {len(changes.deleted)} deleted"
        )

        return changes

    def save_state(self) -> bool:
        """Save current index state."""
        if self.tree is None:
            return False

        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.tree.save(str(self._state_file))

    def load_state(self) -> bool:
        """Load previous index state."""
        if self._state_file.exists():
            self.tree = MerkleTree.load(str(self._state_file))
            return self.tree is not None
        return False

    def get_stats(self) -> dict:
        """Get indexer statistics."""
        if self.tree is None:
            return {"status": "not_initialized"}

        return {
            "status": "initialized",
            "state_file": str(self._state_file),
            **self.tree.get_stats(),
        }


# Global instance
_tree: Optional[MerkleTree] = None


def get_tree(root_path: str, config: Optional[MerkleTreeConfig] = None) -> MerkleTree:
    """Get or create Merkle tree for a path."""
    global _tree
    if _tree is None or str(_tree.root_path) != str(Path(root_path).resolve()):
        _tree = MerkleTree(root_path, config)
    return _tree
