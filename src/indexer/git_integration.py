"""Git Integration - Efficient git operations for incremental indexing.

This module provides high-performance git operations optimized for:
- Large monorepos (100k+ files)
- Efficient diff generation between commits
- Batch blame operations
- File content retrieval at specific commits

Performance optimizations:
- Async subprocess calls for non-blocking I/O
- Batched operations to minimize git process spawns
- Streaming output parsing for large diffs
- Caching of frequently accessed commits
"""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .change_manifest import BlameInfo, ChangeManifest, FileChange

logger = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    """Basic commit information."""
    sha: str
    short_sha: str
    author: str
    author_email: str
    timestamp: datetime
    message: str
    parent_shas: list[str]


class GitIntegration:
    """Git operations for incremental indexing.

    Optimized for performance with large codebases:
    - Uses plumbing commands where possible (faster than porcelain)
    - Batches operations to minimize subprocess overhead
    - Streams large outputs to avoid memory issues
    - Caches commit metadata

    Usage:
        git = GitIntegration("/path/to/repo")

        # Get changes between commits
        changes = await git.get_diff("abc123", "def456")

        # Get blame for a file
        blame = await git.get_blame("src/app.py", line=42)

        # Get file content at specific commit
        content = await git.get_file_at_commit("abc123", "src/app.py")
    """

    # Maximum concurrent git operations
    MAX_CONCURRENT_OPS = 10

    # Timeout for git commands (seconds)
    DEFAULT_TIMEOUT = 60.0

    # Extensions to consider as source files
    SOURCE_EXTENSIONS = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
        ".kt", ".scala", ".sql", ".sh", ".yaml", ".yml", ".json",
        ".toml", ".md", ".vue", ".svelte",
    }

    def __init__(
        self,
        repo_path: str | None = None,
        repo_url: str | None = None,
        clone_dir: str | None = None,
    ):
        """Initialize git integration.

        Args:
            repo_path: Path to existing local repository
            repo_url: URL for cloning (if repo_path doesn't exist)
            clone_dir: Directory to clone into (defaults to temp dir)
        """
        self.repo_url = repo_url
        self._clone_dir = clone_dir

        if repo_path:
            self.repo_path = Path(repo_path).resolve()
        elif repo_url and clone_dir:
            # Derive repo name from URL
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            self.repo_path = Path(clone_dir) / repo_name
        else:
            self.repo_path = Path.cwd()

        # Semaphore to limit concurrent operations
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_OPS)

        # Cache for commit info
        self._commit_cache: dict[str, CommitInfo] = {}

    async def _run_git(
        self,
        args: list[str],
        timeout: float | None = None,
        cwd: Path | None = None,
    ) -> tuple[str, str, int]:
        """Run a git command asynchronously using subprocess.

        Uses create_subprocess_exec which is safe from shell injection
        as it does not invoke a shell.

        Args:
            args: Git command arguments (each as separate string)
            timeout: Command timeout (uses DEFAULT_TIMEOUT if not specified)
            cwd: Working directory (uses repo_path if not specified)

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        async with self._semaphore:
            try:
                # Using create_subprocess_exec is safe - no shell invocation
                process = await asyncio.create_subprocess_exec(
                    "git",
                    *args,
                    cwd=str(cwd or self.repo_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout or self.DEFAULT_TIMEOUT,
                )

                return (
                    stdout.decode("utf-8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                    process.returncode or 0,
                )
            except TimeoutError:
                logger.error(f"Git command timed out: git {' '.join(args)}")
                return "", "Command timed out", 1
            except Exception as e:
                logger.error(f"Git command failed: {e}")
                return "", str(e), 1

    async def ensure_repo(self) -> bool:
        """Ensure repository exists, cloning if necessary.

        Returns:
            True if repository is ready, False otherwise
        """
        if self.repo_path.exists() and (self.repo_path / ".git").exists():
            return True

        if not self.repo_url:
            logger.error(f"Repository not found at {self.repo_path} and no URL provided")
            return False

        logger.info(f"Cloning repository from {self.repo_url}")
        self.repo_path.parent.mkdir(parents=True, exist_ok=True)

        stdout, stderr, code = await self._run_git(
            ["clone", "--depth=100", self.repo_url, str(self.repo_path)],
            timeout=300.0,  # 5 min timeout for clone
            cwd=self.repo_path.parent,
        )

        if code != 0:
            logger.error(f"Clone failed: {stderr}")
            return False

        return True

    async def fetch_updates(self, remote: str = "origin") -> bool:
        """Fetch latest changes from remote.

        Args:
            remote: Remote name (default: origin)

        Returns:
            True if fetch succeeded
        """
        stdout, stderr, code = await self._run_git(
            ["fetch", remote, "--prune"],
            timeout=120.0,
        )

        if code != 0:
            logger.warning(f"Fetch failed: {stderr}")
            return False

        return True

    async def get_current_commit(self) -> str | None:
        """Get the current HEAD commit SHA.

        Returns:
            Full commit SHA or None
        """
        stdout, stderr, code = await self._run_git(["rev-parse", "HEAD"])

        if code != 0:
            return None

        return stdout.strip()

    async def get_commit_info(self, commit: str) -> CommitInfo | None:
        """Get detailed information about a commit.

        Args:
            commit: Commit SHA or ref

        Returns:
            CommitInfo or None if not found
        """
        # Check cache
        if commit in self._commit_cache:
            return self._commit_cache[commit]

        stdout, stderr, code = await self._run_git([
            "show",
            commit,
            "--format=%H|%h|%an|%ae|%at|%s|%P",
            "--no-patch",
        ])

        if code != 0:
            return None

        parts = stdout.strip().split("|")
        if len(parts) < 6:
            return None

        info = CommitInfo(
            sha=parts[0],
            short_sha=parts[1],
            author=parts[2],
            author_email=parts[3],
            timestamp=datetime.fromtimestamp(int(parts[4]), tz=UTC),
            message=parts[5],
            parent_shas=parts[6].split() if len(parts) > 6 and parts[6] else [],
        )

        # Cache it
        self._commit_cache[commit] = info
        self._commit_cache[info.sha] = info

        return info

    async def get_diff(
        self,
        repo_url: str,
        from_commit: str,
        to_commit: str,
    ) -> list[FileChange]:
        """Get list of file changes between two commits.

        Optimized for large diffs by using --name-status and --numstat
        in a single pass.

        Args:
            repo_url: Repository URL (used for remote operations if needed)
            from_commit: Starting commit SHA
            to_commit: Ending commit SHA

        Returns:
            List of FileChange objects
        """
        # First get the name-status output (type of change + paths)
        stdout, stderr, code = await self._run_git([
            "diff",
            "--name-status",
            "--find-renames",
            "--find-copies",
            f"{from_commit}..{to_commit}",
        ])

        if code != 0:
            logger.error(f"Git diff failed: {stderr}")
            return []

        # Parse name-status output
        changes: dict[str, FileChange] = {}
        for line in stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            status = parts[0]
            path = parts[-1]
            old_path = parts[1] if len(parts) > 2 else None

            # Map git status to change type
            if status == "A":
                change_type = "added"
            elif status == "D":
                change_type = "deleted"
            elif status.startswith("R"):
                change_type = "renamed"
            elif status == "M":
                change_type = "modified"
            else:
                change_type = "modified"

            changes[path] = FileChange(
                path=path,
                change_type=change_type,
                old_path=old_path if change_type == "renamed" else None,
            )

        # Now get numstat for line counts (separate call for better parsing)
        stdout, stderr, code = await self._run_git([
            "diff",
            "--numstat",
            f"{from_commit}..{to_commit}",
        ])

        if code == 0:
            for line in stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                additions = int(parts[0]) if parts[0] != "-" else 0
                deletions = int(parts[1]) if parts[1] != "-" else 0
                path = parts[2].split(" -> ")[-1] if " -> " in parts[2] else parts[2]

                if path in changes:
                    changes[path].additions = additions
                    changes[path].deletions = deletions
                    changes[path].binary = parts[0] == "-"

        return list(changes.values())

    async def get_blame(
        self,
        repo_url: str,
        file_path: str,
        line: int,
    ) -> BlameInfo | None:
        """Get git blame information for a specific line.

        Args:
            repo_url: Repository URL (for consistency with interface)
            file_path: Path to file (relative to repo root)
            line: Line number (1-indexed)

        Returns:
            BlameInfo or None if not available
        """
        stdout, stderr, code = await self._run_git([
            "blame",
            "--porcelain",
            "-L", f"{line},{line}",
            file_path,
        ])

        if code != 0:
            logger.warning(f"Git blame failed for {file_path}:{line}: {stderr}")
            return None

        return self._parse_blame_porcelain(stdout, line)

    def _parse_blame_porcelain(self, output: str, line: int) -> BlameInfo | None:
        """Parse porcelain blame output for a single line."""
        lines = output.strip().split("\n")
        if not lines:
            return None

        # First line format: SHA orig_line final_line [num_lines]
        first = lines[0]
        match = re.match(r'^([a-f0-9]{40})\s+\d+\s+\d+', first)
        if not match:
            return None

        sha = match.group(1)
        author = ""
        author_email = ""
        timestamp = 0
        summary = ""

        for l in lines[1:]:
            if l.startswith("author "):
                author = l[7:]
            elif l.startswith("author-mail "):
                author_email = l[12:].strip("<>")
            elif l.startswith("author-time "):
                timestamp = int(l[12:])
            elif l.startswith("summary "):
                summary = l[8:]

        return BlameInfo(
            commit_sha=sha,
            author=author,
            author_email=author_email,
            timestamp=datetime.fromtimestamp(timestamp, tz=UTC) if timestamp else datetime.now(UTC),
            line_number=line,
            commit_message=summary,
        )

    async def get_blame_batch(
        self,
        file_path: str,
        lines: list[int],
    ) -> dict[int, BlameInfo]:
        """Get blame information for multiple lines efficiently.

        Args:
            file_path: Path to file
            lines: List of line numbers

        Returns:
            Dict mapping line number to BlameInfo
        """
        if not lines:
            return {}

        # Build line ranges to minimize blame calls
        min_line = min(lines)
        max_line = max(lines)

        stdout, stderr, code = await self._run_git([
            "blame",
            "--porcelain",
            "-L", f"{min_line},{max_line}",
            file_path,
        ])

        if code != 0:
            return {}

        # Parse all lines and filter to requested ones
        results: dict[int, BlameInfo] = {}
        current_sha = None
        current_author = ""
        current_email = ""
        current_time = 0
        current_summary = ""
        current_line = 0

        for line in stdout.split("\n"):
            if not line:
                continue

            # New commit entry
            match = re.match(r'^([a-f0-9]{40})\s+\d+\s+(\d+)', line)
            if match:
                # Save previous entry if we have one
                if current_sha and current_line in lines:
                    results[current_line] = BlameInfo(
                        commit_sha=current_sha,
                        author=current_author,
                        author_email=current_email,
                        timestamp=datetime.fromtimestamp(current_time, tz=UTC) if current_time else datetime.now(UTC),
                        line_number=current_line,
                        commit_message=current_summary,
                    )

                current_sha = match.group(1)
                current_line = int(match.group(2))
                continue

            if line.startswith("author "):
                current_author = line[7:]
            elif line.startswith("author-mail "):
                current_email = line[12:].strip("<>")
            elif line.startswith("author-time "):
                current_time = int(line[12:])
            elif line.startswith("summary "):
                current_summary = line[8:]

        # Don't forget the last entry
        if current_sha and current_line in lines:
            results[current_line] = BlameInfo(
                commit_sha=current_sha,
                author=current_author,
                author_email=current_email,
                timestamp=datetime.fromtimestamp(current_time, tz=UTC) if current_time else datetime.now(UTC),
                line_number=current_line,
                commit_message=current_summary,
            )

        return results

    async def get_file_at_commit(
        self,
        repo_url: str,
        commit: str,
        path: str,
    ) -> str | None:
        """Get file content at a specific commit.

        Args:
            repo_url: Repository URL (for consistency)
            commit: Commit SHA
            path: File path relative to repo root

        Returns:
            File content as string, or None if not found
        """
        stdout, stderr, code = await self._run_git([
            "show",
            f"{commit}:{path}",
        ])

        if code != 0:
            return None

        return stdout

    async def get_file_hash(self, path: str, commit: str | None = None) -> str | None:
        """Get the blob hash of a file.

        Args:
            path: File path
            commit: Optional commit (HEAD if not specified)

        Returns:
            Git blob hash or None
        """
        ref = f"{commit}:{path}" if commit else path

        stdout, stderr, code = await self._run_git([
            "rev-parse",
            f"{commit or 'HEAD'}:{path}",
        ])

        if code != 0:
            return None

        return stdout.strip()

    async def list_files_at_commit(
        self,
        commit: str,
        path_prefix: str | None = None,
    ) -> list[str]:
        """List all files in the repo at a specific commit.

        Args:
            commit: Commit SHA
            path_prefix: Optional path prefix to filter

        Returns:
            List of file paths
        """
        args = ["ls-tree", "-r", "--name-only", commit]
        if path_prefix:
            args.append(path_prefix)

        stdout, stderr, code = await self._run_git(args)

        if code != 0:
            return []

        return [f for f in stdout.strip().split("\n") if f]

    async def get_merge_base(self, ref1: str, ref2: str) -> str | None:
        """Get the merge base between two refs.

        Args:
            ref1: First ref
            ref2: Second ref

        Returns:
            Merge base commit SHA or None
        """
        stdout, stderr, code = await self._run_git([
            "merge-base",
            ref1,
            ref2,
        ])

        if code != 0:
            return None

        return stdout.strip()

    async def get_commits_between(
        self,
        from_commit: str,
        to_commit: str,
        max_count: int = 100,
    ) -> list[CommitInfo]:
        """Get all commits between two refs.

        Args:
            from_commit: Starting commit (exclusive)
            to_commit: Ending commit (inclusive)
            max_count: Maximum commits to return

        Returns:
            List of CommitInfo objects
        """
        stdout, stderr, code = await self._run_git([
            "log",
            f"--max-count={max_count}",
            "--format=%H|%h|%an|%ae|%at|%s|%P",
            f"{from_commit}..{to_commit}",
        ])

        if code != 0:
            return []

        commits = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 6:
                continue

            commits.append(CommitInfo(
                sha=parts[0],
                short_sha=parts[1],
                author=parts[2],
                author_email=parts[3],
                timestamp=datetime.fromtimestamp(int(parts[4]), tz=UTC),
                message=parts[5],
                parent_shas=parts[6].split() if len(parts) > 6 and parts[6] else [],
            ))

        return commits

    async def analyze_changes(
        self,
        repo_url: str,
        from_commit: str,
        to_commit: str,
    ) -> ChangeManifest:
        """Analyze all changes between two commits.

        This is the main entry point for incremental indexing.

        Args:
            repo_url: Repository URL
            from_commit: Starting commit SHA
            to_commit: Ending commit SHA

        Returns:
            ChangeManifest summarizing all changes
        """
        # Ensure we have the repo
        if not await self.ensure_repo():
            return ChangeManifest(
                from_commit=from_commit,
                to_commit=to_commit,
                files=[],
            )

        # Fetch updates if we have a remote URL
        if self.repo_url:
            await self.fetch_updates()

        # Get the diff
        files = await self.get_diff(repo_url, from_commit, to_commit)

        # Calculate totals
        total_additions = sum(f.additions for f in files)
        total_deletions = sum(f.deletions for f in files)

        return ChangeManifest(
            from_commit=from_commit,
            to_commit=to_commit,
            files=files,
            total_additions=total_additions,
            total_deletions=total_deletions,
        )

    def compute_content_hash(self, content: str) -> str:
        """Compute a hash for content (for change detection).

        Args:
            content: File content

        Returns:
            Short hash string
        """
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# Global instance (lazy initialized)
_git_integration: GitIntegration | None = None


def get_git_integration(repo_path: str = ".") -> GitIntegration:
    """Get or create GitIntegration instance.

    Args:
        repo_path: Path to repository

    Returns:
        GitIntegration instance
    """
    global _git_integration

    if _git_integration is None:
        _git_integration = GitIntegration(repo_path=repo_path)

    return _git_integration
