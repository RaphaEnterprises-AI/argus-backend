"""Tests for the git analyzer service."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


class TestGitCommit:
    """Tests for GitCommit dataclass."""

    def test_git_commit_creation(self):
        """Test creating a GitCommit instance."""
        from src.services.git_analyzer import GitCommit

        commit = GitCommit(
            sha="abc123def456",
            short_sha="abc123d",
            author="John Doe",
            author_email="john@example.com",
            date=datetime.now(UTC),
            message="Fix button styling",
            files_changed=["src/Button.tsx", "src/Button.css"],
        )

        assert commit.sha == "abc123def456"
        assert commit.short_sha == "abc123d"
        assert commit.author == "John Doe"
        assert commit.author_email == "john@example.com"
        assert len(commit.files_changed) == 2

    def test_git_commit_default_files_changed(self):
        """Test GitCommit with default empty files_changed."""
        from src.services.git_analyzer import GitCommit

        commit = GitCommit(
            sha="abc123",
            short_sha="abc",
            author="Jane",
            author_email="jane@example.com",
            date=datetime.now(UTC),
            message="Update",
        )

        assert commit.files_changed == []


class TestGitBlameResult:
    """Tests for GitBlameResult dataclass."""

    def test_git_blame_result_creation(self):
        """Test creating a GitBlameResult instance."""
        from src.services.git_analyzer import GitBlameResult

        result = GitBlameResult(
            commit_sha="abc123def456",
            author="John Doe",
            author_email="john@example.com",
            date=datetime.now(UTC),
            line_number=42,
            line_content='<button data-testid="submit">Submit</button>',
            commit_message="Add submit button",
        )

        assert result.commit_sha == "abc123def456"
        assert result.line_number == 42
        assert "data-testid" in result.line_content
        assert result.commit_message == "Add submit button"

    def test_git_blame_result_optional_message(self):
        """Test GitBlameResult with optional commit message."""
        from src.services.git_analyzer import GitBlameResult

        result = GitBlameResult(
            commit_sha="abc123",
            author="Jane",
            author_email="jane@example.com",
            date=datetime.now(UTC),
            line_number=1,
            line_content="code",
        )

        assert result.commit_message is None


class TestSelectorChange:
    """Tests for SelectorChange dataclass."""

    def test_selector_change_creation(self):
        """Test creating a SelectorChange instance."""
        from src.services.git_analyzer import GitCommit, SelectorChange

        commit = GitCommit(
            sha="abc123",
            short_sha="abc",
            author="John",
            author_email="john@example.com",
            date=datetime.now(UTC),
            message="Rename selector",
        )

        change = SelectorChange(
            old_selector="submit-btn",
            new_selector="submit-button",
            commit=commit,
            file_path="src/Button.tsx",
            line_number=10,
            change_type="renamed",
            context="-data-testid='submit-btn'\n+data-testid='submit-button'",
        )

        assert change.old_selector == "submit-btn"
        assert change.new_selector == "submit-button"
        assert change.change_type == "renamed"
        assert change.commit.sha == "abc123"


class TestCodeChange:
    """Tests for CodeChange dataclass."""

    def test_code_change_creation(self):
        """Test creating a CodeChange instance."""
        from src.services.git_analyzer import CodeChange, GitCommit

        commit = GitCommit(
            sha="abc123",
            short_sha="abc",
            author="John",
            author_email="john@example.com",
            date=datetime.now(UTC),
            message="Update component",
        )

        change = CodeChange(
            file_path="src/Button.tsx",
            commit=commit,
            diff="+ new code\n- old code",
            selectors_affected=["submit-btn", "cancel-btn"],
            components_affected=["Button"],
        )

        assert change.file_path == "src/Button.tsx"
        assert len(change.selectors_affected) == 2
        assert "Button" in change.components_affected


class TestGitAnalyzerInit:
    """Tests for GitAnalyzer initialization."""

    def test_init_with_default_path(self):
        """Test initialization with default path."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()
        assert analyzer.repo_path == Path(".").resolve()

    def test_init_with_custom_path(self):
        """Test initialization with custom path."""
        from src.services.git_analyzer import GitAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = GitAnalyzer(tmpdir)
            assert analyzer.repo_path == Path(tmpdir).resolve()

    def test_validate_repo_logs_warning_if_not_git(self):
        """Test that validation logs warning for non-git directory."""
        from src.services.git_analyzer import GitAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.services.git_analyzer.logger") as mock_logger:
                GitAnalyzer(tmpdir)
                mock_logger.warning.assert_called_once()


class TestGitAnalyzerRunGitCommand:
    """Tests for running git commands."""

    @pytest.mark.asyncio
    async def test_run_git_command_success(self):
        """Test successful git command execution."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"output", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            stdout, stderr, code = await analyzer._run_git_command(["status"])

        assert stdout == "output"
        assert stderr == ""
        assert code == 0

    @pytest.mark.asyncio
    async def test_run_git_command_timeout(self):
        """Test git command timeout handling."""

        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=TimeoutError())
        mock_process.returncode = None

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            stdout, stderr, code = await analyzer._run_git_command(
                ["log"], timeout=0.001
            )

        assert stdout == ""
        assert "timed out" in stderr.lower()
        assert code == 1

    @pytest.mark.asyncio
    async def test_run_git_command_error(self):
        """Test git command error handling."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=Exception("Process failed"),
        ):
            stdout, stderr, code = await analyzer._run_git_command(["status"])

        assert stdout == ""
        assert "Process failed" in stderr
        assert code == 1


class TestGitAnalyzerGetBlame:
    """Tests for git blame functionality."""

    @pytest.mark.asyncio
    async def test_get_blame_for_file(self):
        """Test getting blame for entire file."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        blame_output = """abc123def456789012345678901234567890abcd 1 1 1
author John Doe
author-mail <john@example.com>
author-time 1700000000
author-tz +0000
committer John Doe
committer-mail <john@example.com>
committer-time 1700000000
committer-tz +0000
summary Initial commit
filename test.txt
\tconst x = 1;"""

        with patch.object(
            analyzer, "_run_git_command", return_value=(blame_output, "", 0)
        ):
            results = await analyzer.get_blame("test.txt")

        assert len(results) == 1
        assert results[0].author == "John Doe"
        assert results[0].author_email == "john@example.com"
        assert results[0].line_content == "const x = 1;"

    @pytest.mark.asyncio
    async def test_get_blame_for_specific_line(self):
        """Test getting blame for specific line."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(analyzer, "_run_git_command") as mock_cmd:
            mock_cmd.return_value = ("", "", 0)
            await analyzer.get_blame("test.txt", line_number=10)

        call_args = mock_cmd.call_args[0][0]
        assert "-L" in call_args
        assert "10,10" in call_args

    @pytest.mark.asyncio
    async def test_get_blame_for_line_range(self):
        """Test getting blame for line range."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(analyzer, "_run_git_command") as mock_cmd:
            mock_cmd.return_value = ("", "", 0)
            await analyzer.get_blame("test.txt", line_range=(5, 15))

        call_args = mock_cmd.call_args[0][0]
        assert "-L" in call_args
        assert "5,15" in call_args

    @pytest.mark.asyncio
    async def test_get_blame_handles_error(self):
        """Test that get_blame handles errors gracefully."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(
            analyzer, "_run_git_command", return_value=("", "error", 1)
        ):
            results = await analyzer.get_blame("nonexistent.txt")

        assert results == []


class TestGitAnalyzerParseBlameOutput:
    """Tests for parsing blame output."""

    def test_parse_blame_output_empty(self):
        """Test parsing empty blame output."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()
        results = analyzer._parse_blame_output("")
        assert results == []

    def test_parse_blame_output_multiple_lines(self):
        """Test parsing blame output with multiple lines."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        blame_output = """abc123def456789012345678901234567890abcd 1 1 1
author John
author-mail <john@example.com>
author-time 1700000000
\tline 1
def789abc123456789012345678901234567890ef 2 2 1
author Jane
author-mail <jane@example.com>
author-time 1700000001
\tline 2"""

        results = analyzer._parse_blame_output(blame_output)

        assert len(results) == 2
        assert results[0].author == "John"
        assert results[0].line_content == "line 1"
        assert results[1].author == "Jane"
        assert results[1].line_content == "line 2"


class TestGitAnalyzerGetRecentCommits:
    """Tests for getting recent commits."""

    @pytest.mark.asyncio
    async def test_get_recent_commits(self):
        """Test getting recent commits."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        log_output = """abc123|abc|John Doe|john@example.com|1700000000|Add feature

src/feature.ts
src/feature.test.ts

def456|def|Jane Doe|jane@example.com|1700000001|Fix bug

src/bug.ts"""

        with patch.object(
            analyzer, "_run_git_command", return_value=(log_output, "", 0)
        ):
            commits = await analyzer.get_recent_commits(days=7)

        assert len(commits) == 2
        assert commits[0].author == "John Doe"
        assert commits[0].message == "Add feature"
        assert "src/feature.ts" in commits[0].files_changed
        assert commits[1].author == "Jane Doe"

    @pytest.mark.asyncio
    async def test_get_recent_commits_for_file(self):
        """Test getting recent commits for specific file."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(analyzer, "_run_git_command") as mock_cmd:
            mock_cmd.return_value = ("", "", 0)
            await analyzer.get_recent_commits(file_path="src/Button.tsx")

        call_args = mock_cmd.call_args[0][0]
        assert "--" in call_args
        assert "src/Button.tsx" in call_args

    @pytest.mark.asyncio
    async def test_get_recent_commits_handles_error(self):
        """Test that get_recent_commits handles errors gracefully."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(
            analyzer, "_run_git_command", return_value=("", "error", 1)
        ):
            commits = await analyzer.get_recent_commits()

        assert commits == []


class TestGitAnalyzerParseLogOutput:
    """Tests for parsing log output."""

    def test_parse_log_output_empty(self):
        """Test parsing empty log output."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()
        commits = analyzer._parse_log_output("")
        assert commits == []

    def test_parse_log_output_malformed(self):
        """Test parsing malformed log output."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()
        commits = analyzer._parse_log_output("not|enough|parts")
        assert commits == []

    def test_parse_log_output_no_files(self):
        """Test parsing log output with no files changed."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        log_output = """abc123|abc|John|john@example.com|1700000000|Message"""

        commits = analyzer._parse_log_output(log_output)

        assert len(commits) == 1
        assert commits[0].files_changed == []


class TestGitAnalyzerGetCommitDiff:
    """Tests for getting commit diffs."""

    @pytest.mark.asyncio
    async def test_get_commit_diff(self):
        """Test getting diff for a commit."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        diff_output = """+++ b/src/Button.tsx
--- a/src/Button.tsx
@@ -1,3 +1,3 @@
-<button data-testid="old">
+<button data-testid="new">"""

        with patch.object(
            analyzer, "_run_git_command", return_value=(diff_output, "", 0)
        ):
            diff = await analyzer.get_commit_diff("abc123")

        assert "data-testid" in diff

    @pytest.mark.asyncio
    async def test_get_commit_diff_for_file(self):
        """Test getting diff for specific file."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(analyzer, "_run_git_command") as mock_cmd:
            mock_cmd.return_value = ("", "", 0)
            await analyzer.get_commit_diff("abc123", file_path="src/Button.tsx")

        call_args = mock_cmd.call_args[0][0]
        assert "--" in call_args
        assert "src/Button.tsx" in call_args

    @pytest.mark.asyncio
    async def test_get_commit_diff_handles_error(self):
        """Test that get_commit_diff handles errors gracefully."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(
            analyzer, "_run_git_command", return_value=("", "error", 1)
        ):
            diff = await analyzer.get_commit_diff("nonexistent")

        assert diff == ""


class TestGitAnalyzerFindSelectorChanges:
    """Tests for finding selector changes."""

    @pytest.mark.asyncio
    async def test_find_selector_changes(self):
        """Test finding selector changes in git history."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        log_output = """COMMIT:abc123|abc|John|john@example.com|1700000000|Update selector
diff --git a/src/Button.tsx b/src/Button.tsx
--- a/src/Button.tsx
+++ b/src/Button.tsx
-<button data-testid="old-btn">
+<button data-testid="new-btn">"""

        with patch.object(
            analyzer, "_run_git_command", return_value=(log_output, "", 0)
        ):
            changes = await analyzer.find_selector_changes("new-btn")

        assert len(changes) >= 1

    @pytest.mark.asyncio
    async def test_find_selector_changes_handles_error(self):
        """Test that find_selector_changes handles errors gracefully."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(
            analyzer, "_run_git_command", return_value=("", "error", 1)
        ):
            changes = await analyzer.find_selector_changes("selector")

        assert changes == []


class TestGitAnalyzerParseSelectorChanges:
    """Tests for parsing selector changes."""

    def test_parse_selector_changes_added(self):
        """Test parsing selector added changes."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        output = """COMMIT:abc123|abc|John|john@example.com|1700000000|Add button
diff --git a/src/Button.tsx b/src/Button.tsx
+<button data-testid="submit">"""

        changes = analyzer._parse_selector_changes(output, "submit")

        assert len(changes) == 1
        assert changes[0].change_type == "added"
        assert changes[0].new_selector == "submit"

    def test_parse_selector_changes_removed(self):
        """Test parsing selector removed changes."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        output = """COMMIT:abc123|abc|John|john@example.com|1700000000|Remove button
diff --git a/src/Button.tsx b/src/Button.tsx
-<button data-testid="old-btn">"""

        changes = analyzer._parse_selector_changes(output, "old-btn")

        assert len(changes) == 1
        assert changes[0].change_type == "removed"
        assert changes[0].old_selector == "old-btn"

    def test_parse_selector_changes_empty(self):
        """Test parsing empty output."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()
        changes = analyzer._parse_selector_changes("", "selector")
        assert changes == []


class TestGitAnalyzerFindReplacementSelector:
    """Tests for finding replacement selectors."""

    @pytest.mark.asyncio
    async def test_find_replacement_selector_found(self):
        """Test finding replacement for broken selector."""
        from src.services.git_analyzer import GitAnalyzer, GitCommit, SelectorChange

        analyzer = GitAnalyzer()

        # Mock find_selector_changes to return a removal
        mock_removal = SelectorChange(
            old_selector="old-btn",
            new_selector=None,
            commit=GitCommit(
                sha="abc123",
                short_sha="abc",
                author="John",
                author_email="john@example.com",
                date=datetime.now(UTC),
                message="Rename selector",
            ),
            file_path="src/Button.tsx",
            line_number=10,
            change_type="removed",
            context="context",
        )

        diff_output = """-<button data-testid="old-btn">
+<button data-testid="new-btn">"""

        with patch.object(
            analyzer, "find_selector_changes", return_value=[mock_removal]
        ):
            with patch.object(
                analyzer, "get_commit_diff", return_value=diff_output
            ):
                result = await analyzer.find_replacement_selector("old-btn")

        assert result is not None
        assert result.new_selector == "new-btn"
        assert result.change_type == "renamed"

    @pytest.mark.asyncio
    async def test_find_replacement_selector_not_found(self):
        """Test when no replacement is found."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(analyzer, "find_selector_changes", return_value=[]):
            result = await analyzer.find_replacement_selector("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_find_replacement_selector_no_replacement(self):
        """Test when selector was removed but not replaced."""
        from src.services.git_analyzer import GitAnalyzer, GitCommit, SelectorChange

        analyzer = GitAnalyzer()

        mock_removal = SelectorChange(
            old_selector="deleted-btn",
            new_selector=None,
            commit=GitCommit(
                sha="abc123",
                short_sha="abc",
                author="John",
                author_email="john@example.com",
                date=datetime.now(UTC),
                message="Remove button",
            ),
            file_path="src/Button.tsx",
            line_number=10,
            change_type="removed",
            context="context",
        )

        # Diff without a new selector
        diff_output = """-<button data-testid="deleted-btn">"""

        with patch.object(
            analyzer, "find_selector_changes", return_value=[mock_removal]
        ):
            with patch.object(
                analyzer, "get_commit_diff", return_value=diff_output
            ):
                result = await analyzer.find_replacement_selector("deleted-btn")

        # Returns the removal but without a new_selector
        assert result is not None
        assert result.change_type == "removed"


class TestGitAnalyzerFindReplacementInDiff:
    """Tests for finding replacement in diff."""

    def test_find_replacement_in_diff_found(self):
        """Test finding replacement in diff."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        diff = """-<button data-testid="old-btn">
+<button data-testid="new-btn">"""

        result = analyzer._find_replacement_in_diff(diff, "old-btn")
        assert result == "new-btn"

    def test_find_replacement_in_diff_not_found(self):
        """Test when no replacement is in diff."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        diff = """-<button data-testid="old-btn">"""

        result = analyzer._find_replacement_in_diff(diff, "old-btn")
        assert result is None

    def test_find_replacement_in_diff_no_removal(self):
        """Test when selector was not removed in diff."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        diff = """+<button data-testid="new-btn">"""

        result = analyzer._find_replacement_in_diff(diff, "nonexistent")
        assert result is None


class TestGitAnalyzerIsSimilarSelectorType:
    """Tests for selector type similarity checking."""

    def test_similar_testid_selectors(self):
        """Test detecting similar data-testid selectors."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        assert analyzer._is_similar_selector_type("data-testid=old", "data-testid=new")
        assert analyzer._is_similar_selector_type("testid=old", "testid=new")

    def test_similar_id_selectors(self):
        """Test detecting similar ID selectors."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        assert analyzer._is_similar_selector_type("#old", "#new")
        assert analyzer._is_similar_selector_type("id=old", "id=new")

    def test_similar_class_selectors(self):
        """Test detecting similar class selectors."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        assert analyzer._is_similar_selector_type(".old", ".new")
        assert analyzer._is_similar_selector_type("class=old", "class=new")

    def test_similar_length_selectors(self):
        """Test detecting selectors of similar length."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        assert analyzer._is_similar_selector_type("submit-button", "cancel-button")
        assert not analyzer._is_similar_selector_type("a", "very-long-selector")


class TestGitAnalyzerGetComponentHistory:
    """Tests for getting component history."""

    @pytest.mark.asyncio
    async def test_get_component_history(self):
        """Test getting history for a component."""
        from src.services.git_analyzer import GitAnalyzer, GitCommit

        analyzer = GitAnalyzer()

        mock_commits = [
            GitCommit(
                sha="abc123",
                short_sha="abc",
                author="John",
                author_email="john@example.com",
                date=datetime.now(UTC),
                message="Update Button",
                files_changed=["src/Button.tsx"],
            )
        ]

        mock_diff = """+<button data-testid="new-btn">
-<button data-testid="old-btn">"""

        with patch.object(analyzer, "get_recent_commits", return_value=mock_commits):
            with patch.object(analyzer, "get_commit_diff", return_value=mock_diff):
                history = await analyzer.get_component_history("src/Button.tsx")

        assert len(history) == 1
        assert history[0].file_path == "src/Button.tsx"
        assert len(history[0].selectors_affected) > 0


class TestGitAnalyzerExtractSelectorsFromDiff:
    """Tests for extracting selectors from diff."""

    def test_extract_data_testid(self):
        """Test extracting data-testid selectors."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        diff = """<button data-testid="submit-btn">
<input data-testid='search-input'>"""

        selectors = analyzer._extract_selectors_from_diff(diff)

        assert "submit-btn" in selectors
        assert "search-input" in selectors

    def test_extract_id_selectors(self):
        """Test extracting ID selectors."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        diff = """<button id="main-btn">
#header { color: red; }"""

        selectors = analyzer._extract_selectors_from_diff(diff)

        assert "main-btn" in selectors
        assert "header" in selectors

    def test_extract_playwright_selectors(self):
        """Test extracting Playwright/Cypress selectors."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        diff = """getByTestId("submit")
getByRole("button")
cy.get("#login-form")"""

        selectors = analyzer._extract_selectors_from_diff(diff)

        assert "submit" in selectors
        assert "button" in selectors
        assert "#login-form" in selectors

    def test_extract_empty_diff(self):
        """Test extracting from empty diff."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()
        selectors = analyzer._extract_selectors_from_diff("")
        assert selectors == []


class TestGitAnalyzerGetCommitDetails:
    """Tests for getting commit details."""

    @pytest.mark.asyncio
    async def test_get_commit_details_success(self):
        """Test getting details for a specific commit."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        show_output = """abc123|abc|John Doe|john@example.com|1700000000|Fix bug

src/Button.tsx"""

        with patch.object(
            analyzer, "_run_git_command", return_value=(show_output, "", 0)
        ):
            commit = await analyzer.get_commit_details("abc123")

        assert commit is not None
        assert commit.author == "John Doe"
        assert commit.message == "Fix bug"

    @pytest.mark.asyncio
    async def test_get_commit_details_not_found(self):
        """Test getting details for nonexistent commit."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(
            analyzer, "_run_git_command", return_value=("", "error", 1)
        ):
            commit = await analyzer.get_commit_details("nonexistent")

        assert commit is None


class TestGetGitAnalyzer:
    """Tests for get_git_analyzer factory function."""

    def test_get_git_analyzer_creates_singleton(self):
        """Test that get_git_analyzer creates singleton."""
        import src.services.git_analyzer as module
        from src.services.git_analyzer import get_git_analyzer

        # Reset singleton
        module._git_analyzer = None

        analyzer1 = get_git_analyzer()
        analyzer2 = get_git_analyzer()

        assert analyzer1 is analyzer2

        # Cleanup
        module._git_analyzer = None

    def test_get_git_analyzer_with_path(self):
        """Test get_git_analyzer with custom path."""
        import src.services.git_analyzer as module
        from src.services.git_analyzer import get_git_analyzer

        # Reset singleton
        module._git_analyzer = None

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = get_git_analyzer(tmpdir)
            assert analyzer is not None

        # Cleanup
        module._git_analyzer = None


class TestSelectorPatterns:
    """Tests for selector pattern constants."""

    def test_selector_patterns_exist(self):
        """Test that selector patterns are defined."""
        from src.services.git_analyzer import GitAnalyzer

        assert len(GitAnalyzer.SELECTOR_PATTERNS) > 0

    def test_selector_patterns_compile(self):
        """Test that all selector patterns are valid regex."""
        import re

        from src.services.git_analyzer import GitAnalyzer

        for pattern in GitAnalyzer.SELECTOR_PATTERNS:
            # Should not raise
            re.compile(pattern)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_run_git_command_handles_unicode(self):
        """Test that git commands handle unicode output."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b"Unicode: \xc3\xa9\xc3\xa0\xc3\xb9", b"")
        )
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            stdout, stderr, code = await analyzer._run_git_command(["log"])

        assert code == 0

    def test_parse_blame_output_handles_missing_fields(self):
        """Test parsing blame with missing optional fields."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        # Minimal blame output
        blame_output = """abc123def456789012345678901234567890abcd 1 1
\tcontent"""

        results = analyzer._parse_blame_output(blame_output)
        # Should handle gracefully even with missing fields
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_recent_commits_respects_max(self):
        """Test that max_commits parameter is passed."""
        from src.services.git_analyzer import GitAnalyzer

        analyzer = GitAnalyzer()

        with patch.object(analyzer, "_run_git_command") as mock_cmd:
            mock_cmd.return_value = ("", "", 0)
            await analyzer.get_recent_commits(max_commits=10)

        call_args = mock_cmd.call_args[0][0]
        assert "-n10" in call_args
