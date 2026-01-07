"""Git Analyzer Service - Code-aware healing through git history.

This service provides:
- Git blame for finding who changed code
- Git diff analysis for understanding changes
- Commit history for tracking selector evolution
- Author attribution for accountability
"""

import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GitCommit:
    """Represents a git commit."""
    sha: str
    short_sha: str
    author: str
    author_email: str
    date: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)


@dataclass
class GitBlameResult:
    """Result from git blame for a specific line."""
    commit_sha: str
    author: str
    author_email: str
    date: datetime
    line_number: int
    line_content: str
    commit_message: Optional[str] = None


@dataclass
class SelectorChange:
    """Tracks a selector change in git history."""
    old_selector: Optional[str]
    new_selector: str
    commit: GitCommit
    file_path: str
    line_number: int
    change_type: str  # "added", "removed", "modified", "renamed"
    context: str  # Surrounding code context


@dataclass
class CodeChange:
    """Represents a code change that might affect tests."""
    file_path: str
    commit: GitCommit
    diff: str
    selectors_affected: list[str] = field(default_factory=list)
    components_affected: list[str] = field(default_factory=list)


class GitAnalyzer:
    """Analyzes git history for code-aware test healing.

    This is the secret sauce that makes Argus better than competitors:
    - We don't just guess why a selector broke
    - We READ THE GIT HISTORY to find the exact commit
    - We provide accountability (who changed it)
    - We understand the INTENT behind changes
    """

    # Common selector patterns in various frameworks
    SELECTOR_PATTERNS = [
        # data-testid (React, Vue, etc.)
        r'data-testid=["\']([^"\']+)["\']',
        r'\[data-testid=["\']?([^"\'>\]]+)',
        # id attributes
        r'id=["\']([^"\']+)["\']',
        r'#([a-zA-Z][a-zA-Z0-9_-]+)',
        # class names
        r'class(?:Name)?=["\']([^"\']+)["\']',
        r'\.([a-zA-Z][a-zA-Z0-9_-]+)',
        # aria labels
        r'aria-label=["\']([^"\']+)["\']',
        # name attributes
        r'name=["\']([^"\']+)["\']',
        # placeholder
        r'placeholder=["\']([^"\']+)["\']',
        # Playwright/Cypress selectors
        r'getByTestId\(["\']([^"\']+)["\']',
        r'getByRole\(["\']([^"\']+)["\']',
        r'getByText\(["\']([^"\']+)["\']',
        r'getByLabel\(["\']([^"\']+)["\']',
        r'locator\(["\']([^"\']+)["\']',
        r'page\.click\(["\']([^"\']+)["\']',
        r'cy\.get\(["\']([^"\']+)["\']',
    ]

    def __init__(self, repo_path: str = "."):
        """Initialize with repository path.

        Args:
            repo_path: Path to git repository root
        """
        self.repo_path = Path(repo_path).resolve()
        self._validate_repo()

    def _validate_repo(self) -> None:
        """Validate that the path is a git repository."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            logger.warning(f"Not a git repository: {self.repo_path}")

    async def _run_git_command(
        self,
        args: list[str],
        timeout: float = 30.0
    ) -> tuple[str, str, int]:
        """Run a git command asynchronously.

        Args:
            args: Git command arguments
            timeout: Command timeout in seconds

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return (
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
                process.returncode or 0
            )
        except asyncio.TimeoutError:
            logger.error(f"Git command timed out: git {' '.join(args)}")
            return "", "Command timed out", 1
        except Exception as e:
            logger.error(f"Git command failed: {e}")
            return "", str(e), 1

    async def get_blame(
        self,
        file_path: str,
        line_number: Optional[int] = None,
        line_range: Optional[tuple[int, int]] = None
    ) -> list[GitBlameResult]:
        """Get git blame for a file or specific lines.

        Args:
            file_path: Path to file (relative to repo)
            line_number: Specific line to blame
            line_range: Range of lines (start, end) to blame

        Returns:
            List of blame results
        """
        args = ["blame", "--porcelain"]

        if line_number:
            args.extend(["-L", f"{line_number},{line_number}"])
        elif line_range:
            args.extend(["-L", f"{line_range[0]},{line_range[1]}"])

        args.append(file_path)

        stdout, stderr, code = await self._run_git_command(args)

        if code != 0:
            logger.warning(f"Git blame failed for {file_path}: {stderr}")
            return []

        return self._parse_blame_output(stdout)

    def _parse_blame_output(self, output: str) -> list[GitBlameResult]:
        """Parse porcelain blame output."""
        results = []
        lines = output.strip().split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # First line of each block is: SHA original_line final_line [num_lines]
            match = re.match(r'^([a-f0-9]{40})\s+(\d+)\s+(\d+)', line)
            if not match:
                i += 1
                continue

            sha = match.group(1)
            line_num = int(match.group(3))

            # Parse metadata lines
            author = ""
            author_email = ""
            author_time = 0

            i += 1
            while i < len(lines) and not lines[i].startswith("\t"):
                meta_line = lines[i]
                if meta_line.startswith("author "):
                    author = meta_line[7:]
                elif meta_line.startswith("author-mail "):
                    author_email = meta_line[12:].strip("<>")
                elif meta_line.startswith("author-time "):
                    author_time = int(meta_line[12:])
                i += 1

            # Get the actual line content (starts with tab)
            content = ""
            if i < len(lines) and lines[i].startswith("\t"):
                content = lines[i][1:]
                i += 1

            results.append(GitBlameResult(
                commit_sha=sha,
                author=author,
                author_email=author_email,
                date=datetime.fromtimestamp(author_time, tz=timezone.utc) if author_time else datetime.now(timezone.utc),
                line_number=line_num,
                line_content=content,
            ))

        return results

    async def get_recent_commits(
        self,
        file_path: Optional[str] = None,
        days: int = 7,
        max_commits: int = 50
    ) -> list[GitCommit]:
        """Get recent commits, optionally for a specific file.

        Args:
            file_path: Optional file path to filter commits
            days: Number of days to look back
            max_commits: Maximum commits to return

        Returns:
            List of commits
        """
        args = [
            "log",
            f"--since={days}.days.ago",
            f"-n{max_commits}",
            "--pretty=format:%H|%h|%an|%ae|%at|%s",
            "--name-only"
        ]

        if file_path:
            args.extend(["--", file_path])

        stdout, stderr, code = await self._run_git_command(args)

        if code != 0:
            logger.warning(f"Git log failed: {stderr}")
            return []

        return self._parse_log_output(stdout)

    def _parse_log_output(self, output: str) -> list[GitCommit]:
        """Parse git log output."""
        commits = []
        blocks = output.strip().split("\n\n")

        for block in blocks:
            if not block.strip():
                continue

            lines = block.strip().split("\n")
            if not lines:
                continue

            # First line is metadata
            parts = lines[0].split("|")
            if len(parts) < 6:
                continue

            # Remaining lines are changed files
            files_changed = [l.strip() for l in lines[1:] if l.strip()]

            commits.append(GitCommit(
                sha=parts[0],
                short_sha=parts[1],
                author=parts[2],
                author_email=parts[3],
                date=datetime.fromtimestamp(int(parts[4]), tz=timezone.utc),
                message=parts[5],
                files_changed=files_changed,
            ))

        return commits

    async def get_commit_diff(
        self,
        commit_sha: str,
        file_path: Optional[str] = None
    ) -> str:
        """Get the diff for a specific commit.

        Args:
            commit_sha: Commit SHA
            file_path: Optional file to filter diff

        Returns:
            Diff string
        """
        args = ["show", "--format=", commit_sha]

        if file_path:
            args.extend(["--", file_path])

        stdout, stderr, code = await self._run_git_command(args)

        if code != 0:
            logger.warning(f"Git show failed for {commit_sha}: {stderr}")
            return ""

        return stdout

    async def find_selector_changes(
        self,
        selector: str,
        days: int = 30,
        search_pattern: Optional[str] = None
    ) -> list[SelectorChange]:
        """Find commits that changed a specific selector.

        This is the KEY FEATURE for code-aware healing:
        - Find when/who changed a selector
        - Understand what it changed to
        - Provide the exact commit for accountability

        Args:
            selector: The selector to search for
            days: Number of days to look back
            search_pattern: Optional regex pattern override

        Returns:
            List of selector changes found
        """
        # Search for the selector in git log
        pattern = search_pattern or re.escape(selector)

        args = [
            "log",
            f"--since={days}.days.ago",
            "-p",  # Show patches
            "--all",
            f"-S{selector}",  # Search for selector changes
            "--pretty=format:COMMIT:%H|%h|%an|%ae|%at|%s"
        ]

        stdout, stderr, code = await self._run_git_command(args, timeout=60.0)

        if code != 0:
            logger.warning(f"Git log -S failed: {stderr}")
            return []

        return self._parse_selector_changes(stdout, selector)

    def _parse_selector_changes(
        self,
        output: str,
        selector: str
    ) -> list[SelectorChange]:
        """Parse git log -p output to find selector changes."""
        changes = []
        current_commit = None
        current_file = None

        lines = output.split("\n")

        for i, line in enumerate(lines):
            # New commit
            if line.startswith("COMMIT:"):
                parts = line[7:].split("|")
                if len(parts) >= 6:
                    current_commit = GitCommit(
                        sha=parts[0],
                        short_sha=parts[1],
                        author=parts[2],
                        author_email=parts[3],
                        date=datetime.fromtimestamp(int(parts[4]), tz=timezone.utc),
                        message=parts[5],
                    )

            # New file in diff
            elif line.startswith("diff --git"):
                match = re.search(r'b/(.+)$', line)
                if match:
                    current_file = match.group(1)

            # Found selector in added line
            elif line.startswith("+") and not line.startswith("+++"):
                if selector in line and current_commit and current_file:
                    # Get context (surrounding lines)
                    start = max(0, i - 3)
                    end = min(len(lines), i + 4)
                    context = "\n".join(lines[start:end])

                    changes.append(SelectorChange(
                        old_selector=None,
                        new_selector=selector,
                        commit=current_commit,
                        file_path=current_file,
                        line_number=i,
                        change_type="added",
                        context=context,
                    ))

            # Found selector in removed line
            elif line.startswith("-") and not line.startswith("---"):
                if selector in line and current_commit and current_file:
                    start = max(0, i - 3)
                    end = min(len(lines), i + 4)
                    context = "\n".join(lines[start:end])

                    changes.append(SelectorChange(
                        old_selector=selector,
                        new_selector=None,
                        commit=current_commit,
                        file_path=current_file,
                        line_number=i,
                        change_type="removed",
                        context=context,
                    ))

        return changes

    async def find_replacement_selector(
        self,
        broken_selector: str,
        file_path: Optional[str] = None,
        days: int = 14
    ) -> Optional[SelectorChange]:
        """Find what a broken selector was replaced with.

        THE MAGIC METHOD: When a selector breaks, this finds:
        1. The commit that removed it
        2. What it was replaced with (if anything)
        3. Who made the change

        Args:
            broken_selector: The selector that's no longer working
            file_path: Optional file path hint
            days: Days to search back

        Returns:
            SelectorChange with new_selector if found
        """
        # First, find when the selector was removed
        changes = await self.find_selector_changes(broken_selector, days=days)

        removed_changes = [c for c in changes if c.change_type == "removed"]

        if not removed_changes:
            logger.info(f"Selector '{broken_selector}' not found in recent git history")
            return None

        # Get the most recent removal
        removal = removed_changes[0]

        # Now get the full diff of that commit to find the replacement
        diff = await self.get_commit_diff(removal.commit.sha, removal.file_path)

        # Look for a new selector added near where the old one was removed
        new_selector = self._find_replacement_in_diff(diff, broken_selector)

        if new_selector:
            return SelectorChange(
                old_selector=broken_selector,
                new_selector=new_selector,
                commit=removal.commit,
                file_path=removal.file_path,
                line_number=removal.line_number,
                change_type="renamed",
                context=removal.context,
            )

        return removal

    def _find_replacement_in_diff(
        self,
        diff: str,
        old_selector: str
    ) -> Optional[str]:
        """Find a replacement selector in a diff.

        Heuristics:
        1. Look for similar patterns (same type of selector)
        2. Look for additions near the removal
        3. Use selector patterns to identify valid candidates
        """
        lines = diff.split("\n")
        removed_line_idx = None

        # Find where the old selector was removed
        for i, line in enumerate(lines):
            if line.startswith("-") and old_selector in line:
                removed_line_idx = i
                break

        if removed_line_idx is None:
            return None

        # Look for additions within 5 lines
        search_range = range(
            max(0, removed_line_idx - 5),
            min(len(lines), removed_line_idx + 6)
        )

        for i in search_range:
            line = lines[i]
            if not line.startswith("+"):
                continue

            # Extract selectors from added line
            for pattern in self.SELECTOR_PATTERNS:
                matches = re.findall(pattern, line)
                for match in matches:
                    # Skip if it's the same selector
                    if match == old_selector:
                        continue

                    # Check if it's a similar type of selector
                    if self._is_similar_selector_type(old_selector, match):
                        return match

        return None

    def _is_similar_selector_type(self, old: str, new: str) -> bool:
        """Check if two selectors are of similar type."""
        # Both are data-testid style
        if "data-testid" in old or "testid" in old.lower():
            return "testid" in new.lower() or "data-" in new

        # Both are ID selectors
        if old.startswith("#") or "id=" in old:
            return new.startswith("#") or "id=" in new

        # Both are class selectors
        if old.startswith(".") or "class" in old:
            return new.startswith(".") or "class" in new

        # Both are similar length (likely same pattern)
        return abs(len(old) - len(new)) < 10

    async def get_component_history(
        self,
        component_path: str,
        days: int = 30
    ) -> list[CodeChange]:
        """Get change history for a component file.

        Useful for understanding the evolution of a component
        and predicting future issues.

        Args:
            component_path: Path to component file
            days: Days to look back

        Returns:
            List of code changes
        """
        commits = await self.get_recent_commits(component_path, days=days)

        changes = []
        for commit in commits:
            diff = await self.get_commit_diff(commit.sha, component_path)

            # Extract selectors affected in this change
            selectors = self._extract_selectors_from_diff(diff)

            changes.append(CodeChange(
                file_path=component_path,
                commit=commit,
                diff=diff,
                selectors_affected=selectors,
            ))

        return changes

    def _extract_selectors_from_diff(self, diff: str) -> list[str]:
        """Extract all selectors mentioned in a diff."""
        selectors = set()

        for pattern in self.SELECTOR_PATTERNS:
            matches = re.findall(pattern, diff)
            selectors.update(matches)

        return list(selectors)

    async def get_commit_details(self, commit_sha: str) -> Optional[GitCommit]:
        """Get details for a specific commit.

        Args:
            commit_sha: Full or short SHA

        Returns:
            GitCommit or None
        """
        args = [
            "show",
            commit_sha,
            "--pretty=format:%H|%h|%an|%ae|%at|%s",
            "--name-only",
            "-q"  # Quiet (no diff)
        ]

        stdout, stderr, code = await self._run_git_command(args)

        if code != 0:
            return None

        commits = self._parse_log_output(stdout)
        return commits[0] if commits else None


# Global instance (lazy initialized)
_git_analyzer: Optional[GitAnalyzer] = None


def get_git_analyzer(repo_path: str = ".") -> GitAnalyzer:
    """Get or create GitAnalyzer instance."""
    global _git_analyzer

    if _git_analyzer is None:
        _git_analyzer = GitAnalyzer(repo_path)

    return _git_analyzer
