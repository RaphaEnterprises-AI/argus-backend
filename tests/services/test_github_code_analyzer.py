"""Comprehensive tests for GitHubCodeAnalyzer including optimized methods.

This module tests the GitHubCodeAnalyzer service which provides code-aware
test healing via the GitHub REST API. It includes tests for:

- URL parsing and validation
- API call tracking (RAP-349)
- Selector pattern matching
- Single and bulk diff parsing (RAP-347)
- GitHub Compare API integration (RAP-346)
- find_replacement_selector optimization (RAP-348)
- Factory function for project configuration
- Error handling and edge cases
"""

import pytest
import asyncio
import inspect
import re
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta


# ============================================================================
# Import Tests
# ============================================================================


class TestImports:
    """Test all imports work correctly."""

    def test_main_imports(self):
        """Test main module imports."""
        from src.services.github_code_analyzer import (
            GitHubCodeAnalyzer,
            GitHubCommit,
            SelectorChange,
            get_github_analyzer_for_project,
        )

        assert GitHubCodeAnalyzer is not None
        assert GitHubCommit is not None
        assert SelectorChange is not None
        assert get_github_analyzer_for_project is not None


# ============================================================================
# URL Parsing Tests
# ============================================================================


class TestURLParsing:
    """Test GitHub URL parsing."""

    def test_standard_https_url(self):
        """Test standard HTTPS URL parsing."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/myorg/myrepo", "token")
        assert analyzer.owner == "myorg"
        assert analyzer.repo == "myrepo"

    def test_url_with_git_suffix(self):
        """Test URL with .git suffix is handled."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer(
            "https://github.com/myorg/myrepo.git", "token"
        )
        assert analyzer.owner == "myorg"
        assert analyzer.repo == "myrepo"

    def test_url_with_trailing_slash(self):
        """Test URL with trailing slash."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/myorg/myrepo/", "token")
        assert analyzer.owner == "myorg"
        assert analyzer.repo == "myrepo"

    def test_url_with_nested_path(self):
        """Test URL with nested path (only first two parts used)."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer(
            "https://github.com/myorg/myrepo/tree/main", "token"
        )
        assert analyzer.owner == "myorg"
        assert analyzer.repo == "myrepo"

    def test_invalid_url_missing_repo(self):
        """Test invalid URL with missing repo raises error."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            GitHubCodeAnalyzer("https://github.com/onlyowner", "token")

    def test_invalid_url_empty_path(self):
        """Test invalid URL with empty path raises error."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            GitHubCodeAnalyzer("https://github.com/", "token")

    def test_github_enterprise_url(self):
        """Test GitHub Enterprise URL parsing."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer(
            "https://github.mycompany.com/internal/myrepo",
            "token",
            api_base="https://api.github.mycompany.com",
        )
        assert analyzer.owner == "internal"
        assert analyzer.repo == "myrepo"
        assert analyzer.api_base == "https://api.github.mycompany.com"


# ============================================================================
# API Call Tracking Tests (RAP-349)
# ============================================================================


class TestAPICallTracking:
    """Test API call counting functionality."""

    def test_initial_count_is_zero(self):
        """Test initial API call count is zero."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")
        assert analyzer.api_call_count == 0

    def test_reset_count(self):
        """Test reset_api_call_count works."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")
        analyzer._api_call_count = 10
        analyzer.reset_api_call_count()
        assert analyzer.api_call_count == 0

    @pytest.mark.asyncio
    async def test_api_call_increments_count(self):
        """Test API calls increment the counter."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        # Mock the client to return a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        analyzer._client = mock_client

        # Make a request
        await analyzer._api_request("/repos/org/repo/commits")

        assert analyzer.api_call_count == 1

        # Make another request
        await analyzer._api_request("/repos/org/repo/commits")

        assert analyzer.api_call_count == 2


# ============================================================================
# Selector Pattern Tests
# ============================================================================


class TestSelectorPatterns:
    """Test selector pattern matching."""

    def test_selector_patterns_defined(self):
        """Test SELECTOR_PATTERNS is defined and non-empty."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        assert hasattr(GitHubCodeAnalyzer, "SELECTOR_PATTERNS")
        assert len(GitHubCodeAnalyzer.SELECTOR_PATTERNS) > 0

    def test_data_testid_pattern(self):
        """Test data-testid pattern matching."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        patterns = GitHubCodeAnalyzer.SELECTOR_PATTERNS
        test_line = '<button data-testid="login-button">Login</button>'

        found = False
        for pattern in patterns:
            matches = re.findall(pattern, test_line)
            if "login-button" in matches:
                found = True
                break

        assert found, "data-testid pattern should match"

    def test_id_attribute_pattern(self):
        """Test id attribute pattern matching."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        patterns = GitHubCodeAnalyzer.SELECTOR_PATTERNS
        test_line = '<input id="email-input" type="email">'

        found = False
        for pattern in patterns:
            matches = re.findall(pattern, test_line)
            if "email-input" in matches:
                found = True
                break

        assert found, "id pattern should match"

    def test_class_pattern(self):
        """Test class attribute pattern matching."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        patterns = GitHubCodeAnalyzer.SELECTOR_PATTERNS
        test_line = '<div className="my-component">'

        found = False
        for pattern in patterns:
            matches = re.findall(pattern, test_line)
            if "my-component" in matches:
                found = True
                break

        assert found, "className pattern should match"


# ============================================================================
# find_selector_in_diff Tests
# ============================================================================


class TestFindSelectorInDiff:
    """Test the find_selector_in_diff method."""

    @pytest.mark.asyncio
    async def test_find_exact_id_change(self):
        """Test finding exact ID selector match in removed line."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """
diff --git a/src/login.tsx b/src/login.tsx
@@ -10,7 +10,7 @@
 export function LoginButton() {
   return (
-    <button id="login-btn">
+    <button id="login-button">
       Login
     </button>
   )
"""

        old_sel, new_sel, line_num = await analyzer.find_selector_in_diff(
            diff, "login-btn"
        )

        # Should find exact match in removed line
        assert old_sel == "login-btn", f"Expected 'login-btn', got {old_sel}"

    @pytest.mark.asyncio
    async def test_find_partial_id_change(self):
        """Test finding ID selector change with substring matching."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """
diff --git a/src/login.tsx b/src/login.tsx
@@ -10,7 +10,7 @@
-    <button id="login-old">
+    <button id="login-new">
"""

        old_sel, new_sel, line_num = await analyzer.find_selector_in_diff(
            diff, "login"
        )

        # Both should match since "login" is a substring of both
        assert old_sel == "login-old"
        assert new_sel == "login-new"

    @pytest.mark.asyncio
    async def test_find_data_testid_change(self):
        """Test finding data-testid exact match in removed line."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """
diff --git a/src/form.tsx b/src/form.tsx
@@ -5,7 +5,7 @@
 export function Form() {
   return (
-    <input data-testid="email-field" />
+    <input data-testid="email-input" />
   )
"""

        old_sel, new_sel, line_num = await analyzer.find_selector_in_diff(
            diff, "email-field"
        )

        # Should find the old selector exactly
        assert old_sel == "email-field"

    @pytest.mark.asyncio
    async def test_find_data_testid_change_with_partial(self):
        """Test finding data-testid change with partial matching."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """
diff --git a/src/form.tsx b/src/form.tsx
@@ -5,7 +5,7 @@
 export function Form() {
   return (
-    <input data-testid="email-field" />
+    <input data-testid="email-new-field" />
   )
"""

        old_sel, new_sel, line_num = await analyzer.find_selector_in_diff(
            diff, "email"
        )

        # Both should match since "email" is a substring of both
        assert old_sel is not None
        assert new_sel is not None

    @pytest.mark.asyncio
    async def test_no_selector_found(self):
        """Test when no matching selector is found."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """
diff --git a/src/unrelated.tsx b/src/unrelated.tsx
@@ -1,3 +1,3 @@
-const foo = "bar";
+const foo = "baz";
"""

        old_sel, new_sel, line_num = await analyzer.find_selector_in_diff(
            diff, "login-btn"
        )

        assert old_sel is None
        assert new_sel is None

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test partial matching of selectors."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """
diff --git a/src/button.tsx b/src/button.tsx
@@ -1,3 +1,3 @@
-    <button id="submit-btn">
+    <button id="submit-button">
"""

        # Search for "submit" - should match both since "submit" is substring
        old_sel, new_sel, line_num = await analyzer.find_selector_in_diff(
            diff, "submit"
        )

        assert old_sel == "submit-btn"
        assert new_sel == "submit-button"


# ============================================================================
# _find_selector_in_bulk_diff Tests (RAP-347)
# ============================================================================


class TestFindSelectorInBulkDiff:
    """Test the bulk diff parsing method."""

    def test_method_exists(self):
        """Test _find_selector_in_bulk_diff method exists."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")
        assert hasattr(analyzer, "_find_selector_in_bulk_diff")

    def test_single_file_diff_with_partial_match(self):
        """Test parsing single file in bulk diff with partial matching."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """diff --git a/src/login.tsx b/src/login.tsx
--- a/src/login.tsx
+++ b/src/login.tsx
@@ -10,7 +10,7 @@
 export function LoginButton() {
   return (
-    <button id="login-old">
+    <button id="login-new">
       Login
     </button>
   )
"""

        result = analyzer._find_selector_in_bulk_diff(diff, "login")

        assert result is not None
        assert result.old_selector == "login-old"
        assert result.new_selector == "login-new"
        assert result.file_path == "src/login.tsx"

    def test_multi_file_diff_with_partial_match(self):
        """Test parsing multiple files in bulk diff."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """diff --git a/src/header.tsx b/src/header.tsx
--- a/src/header.tsx
+++ b/src/header.tsx
@@ -1,3 +1,3 @@
-const logo = "old";
+const logo = "new";
diff --git a/src/login.tsx b/src/login.tsx
--- a/src/login.tsx
+++ b/src/login.tsx
@@ -10,7 +10,7 @@
-    <button data-testid="submit-old">
+    <button data-testid="submit-new">
"""

        result = analyzer._find_selector_in_bulk_diff(diff, "submit")

        assert result is not None
        assert result.old_selector == "submit-old"
        assert result.new_selector == "submit-new"
        assert result.file_path == "src/login.tsx"

    def test_file_hint_filtering(self):
        """Test file hint filters to relevant files."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """diff --git a/src/wrong.tsx b/src/wrong.tsx
--- a/src/wrong.tsx
+++ b/src/wrong.tsx
@@ -10,7 +10,7 @@
-    <button id="btn-old">
+    <button id="btn-new">
diff --git a/src/login.tsx b/src/login.tsx
--- a/src/login.tsx
+++ b/src/login.tsx
@@ -10,7 +10,7 @@
-    <button id="login-old">
+    <button id="login-new">
"""

        result = analyzer._find_selector_in_bulk_diff(
            diff, "login", file_hint="login.tsx"
        )

        assert result is not None
        assert result.file_path == "src/login.tsx"

    def test_file_hint_partial_match(self):
        """Test file hint allows partial path matching."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """diff --git a/src/components/Login.tsx b/src/components/Login.tsx
--- a/src/components/Login.tsx
+++ b/src/components/Login.tsx
@@ -10,7 +10,7 @@
-    <input id="email-old">
+    <input id="email-new">
"""

        result = analyzer._find_selector_in_bulk_diff(
            diff, "email", file_hint="Login.tsx"
        )

        assert result is not None
        assert "Login.tsx" in result.file_path

    def test_no_match_returns_none(self):
        """Test returns None when no match found."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """diff --git a/src/unrelated.tsx b/src/unrelated.tsx
--- a/src/unrelated.tsx
+++ b/src/unrelated.tsx
@@ -1,3 +1,3 @@
-const foo = "bar";
+const foo = "baz";
"""

        result = analyzer._find_selector_in_bulk_diff(diff, "nonexistent-selector")

        assert result is None

    def test_empty_diff(self):
        """Test empty diff returns None."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        result = analyzer._find_selector_in_bulk_diff("", "selector")

        assert result is None

    def test_only_addition_no_removal(self):
        """Test when selector is only added (not modified)."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """diff --git a/src/new.tsx b/src/new.tsx
--- /dev/null
+++ b/src/new.tsx
@@ -0,0 +1,5 @@
+export function NewButton() {
+  return (
+    <button id="brand-new-btn">New</button>
+  )
+}
"""

        result = analyzer._find_selector_in_bulk_diff(diff, "brand-new")

        assert result is not None
        assert result.new_selector == "brand-new-btn"
        assert result.old_selector is None


# ============================================================================
# get_compare_diff Tests (RAP-346)
# ============================================================================


class TestGetCompareDiff:
    """Test the Compare API integration."""

    def test_method_exists(self):
        """Test get_compare_diff method exists."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")
        assert hasattr(analyzer, "get_compare_diff")
        assert inspect.iscoroutinefunction(analyzer.get_compare_diff)

    @pytest.mark.asyncio
    async def test_resolves_head_notation(self):
        """Test HEAD~N notation is resolved to commit SHA."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        # Mock responses
        commits_response = MagicMock()
        commits_response.status_code = 200
        commits_response.json.return_value = [
            {
                "sha": f"commit_{i}",
                "commit": {
                    "author": {
                        "name": "Test",
                        "email": "test@test.com",
                        "date": "2024-01-01T00:00:00Z",
                    },
                    "message": f"Commit {i}",
                },
            }
            for i in range(35)
        ]

        repo_response = MagicMock()
        repo_response.status_code = 200
        repo_response.json.return_value = {"default_branch": "main"}

        diff_response = MagicMock()
        diff_response.status_code = 200
        diff_response.text = "diff --git a/file.txt b/file.txt\n+test"

        async def mock_get(*args, **kwargs):
            url = args[0]

            if "/commits" in url and "/compare/" not in url:
                return commits_response
            elif "compare" in url:
                return diff_response
            else:
                return repo_response

        mock_client = MagicMock()
        mock_client.get = mock_get
        analyzer._client = mock_client

        result = await analyzer.get_compare_diff(base="HEAD~30", head="HEAD")

        # Should have made API calls
        assert analyzer.api_call_count >= 1

    @pytest.mark.asyncio
    async def test_explicit_commit_shas(self):
        """Test explicit commit SHAs work directly."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff_response = MagicMock()
        diff_response.status_code = 200
        diff_response.text = "diff --git a/file.txt b/file.txt\n-old\n+new"

        async def mock_get(*args, **kwargs):
            return diff_response

        mock_client = MagicMock()
        mock_client.get = mock_get
        analyzer._client = mock_client

        result = await analyzer.get_compare_diff(base="abc123", head="def456")

        assert result is not None
        assert "diff --git" in result


# ============================================================================
# find_replacement_selector Tests (RAP-348 - Main Entry Point)
# ============================================================================


class TestFindReplacementSelector:
    """Test the main entry point for code-aware healing."""

    @pytest.mark.asyncio
    async def test_uses_bulk_diff_first(self):
        """Test that bulk diff is tried first (optimization)."""
        from src.services.github_code_analyzer import (
            GitHubCodeAnalyzer,
            SelectorChange,
            GitHubCommit,
        )

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        # Create a mock SelectorChange to return from bulk diff
        mock_commit = GitHubCommit(
            sha="bulk_compare",
            short_sha="bulk",
            author="Test",
            author_email="test@test.com",
            date=datetime.now(timezone.utc),
            message="Bulk compare",
        )
        mock_change = SelectorChange(
            old_selector="old-btn",
            new_selector="new-btn",
            commit=mock_commit,
            file_path="src/file.tsx",
            line_number=10,
            change_type="modified",
            context="<button>...</button>",
        )

        with patch.object(
            analyzer, "get_compare_diff", new_callable=AsyncMock
        ) as mock_compare:
            mock_compare.return_value = "diff --git a/src/file.tsx..."

            with patch.object(analyzer, "_find_selector_in_bulk_diff") as mock_bulk:
                mock_bulk.return_value = mock_change

                with patch.object(
                    analyzer, "get_recent_commits", new_callable=AsyncMock
                ) as mock_commits:
                    mock_commits.return_value = []

                    result = await analyzer.find_replacement_selector("old-btn")

                    # Bulk diff should have been called
                    mock_compare.assert_called_once()

                    # Bulk parser should have been called
                    mock_bulk.assert_called_once()

                    assert result is not None
                    assert result.new_selector == "new-btn"

    @pytest.mark.asyncio
    async def test_falls_back_to_per_commit(self):
        """Test fallback to per-commit when bulk diff fails."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer, GitHubCommit

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        # Mock get_compare_diff to return None (failure)
        with patch.object(
            analyzer, "get_compare_diff", new_callable=AsyncMock
        ) as mock_compare:
            mock_compare.return_value = None

            # Mock get_recent_commits
            mock_commits_list = [
                GitHubCommit(
                    sha="abc123",
                    short_sha="abc123",
                    author="Test",
                    author_email="test@test.com",
                    date=datetime.now(timezone.utc),
                    message="Fix button",
                )
            ]

            with patch.object(
                analyzer, "get_recent_commits", new_callable=AsyncMock
            ) as mock_get_commits:
                mock_get_commits.return_value = mock_commits_list

                # Mock get_commit_diff - use partial matching pattern
                commit_diff = """diff --git a/src/login.tsx b/src/login.tsx
@@ -10,7 +10,7 @@
-    <button id="fallback-old">
+    <button id="fallback-new">
"""

                with patch.object(
                    analyzer, "get_commit_diff", new_callable=AsyncMock
                ) as mock_diff:
                    mock_diff.return_value = commit_diff

                    result = await analyzer.find_replacement_selector("fallback")

                    # Bulk diff was called but returned None
                    mock_compare.assert_called_once()

                    # Fallback path was used
                    mock_get_commits.assert_called_once()
                    mock_diff.assert_called()

                    assert result is not None
                    assert result.new_selector == "fallback-new"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        """Test returns None when selector not found anywhere."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        with patch.object(
            analyzer, "get_compare_diff", new_callable=AsyncMock
        ) as mock_compare:
            mock_compare.return_value = "diff --git a/unrelated.txt\n+nothing here"

            with patch.object(analyzer, "_find_selector_in_bulk_diff") as mock_bulk:
                mock_bulk.return_value = None

                with patch.object(
                    analyzer, "get_recent_commits", new_callable=AsyncMock
                ) as mock_commits:
                    mock_commits.return_value = []

                    result = await analyzer.find_replacement_selector(
                        "nonexistent-selector"
                    )

                    assert result is None

    @pytest.mark.asyncio
    async def test_file_hint_passed_to_bulk_diff(self):
        """Test file hint is passed to bulk diff parsing."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        bulk_diff = """diff --git a/src/login.tsx b/src/login.tsx
--- a/src/login.tsx
+++ b/src/login.tsx
@@ -10,7 +10,7 @@
-    <button id="hint-old">
+    <button id="hint-new">
"""

        with patch.object(
            analyzer, "get_compare_diff", new_callable=AsyncMock
        ) as mock_compare:
            mock_compare.return_value = bulk_diff

            with patch.object(analyzer, "_find_selector_in_bulk_diff") as mock_bulk:
                mock_bulk.return_value = None  # Force fallback

                with patch.object(
                    analyzer, "get_recent_commits", new_callable=AsyncMock
                ) as mock_commits:
                    mock_commits.return_value = []

                    await analyzer.find_replacement_selector(
                        "hint-old", file_hint="login.tsx"
                    )

                    # Check file_hint was passed to bulk diff parser
                    mock_bulk.assert_called_once()
                    call_args = mock_bulk.call_args
                    assert (
                        call_args[1].get("file_hint") == "login.tsx"
                        or call_args[0][2] == "login.tsx"
                    )


# ============================================================================
# Context Extraction Tests
# ============================================================================


class TestContextExtraction:
    """Test _extract_context_from_diff method."""

    def test_extracts_surrounding_lines(self):
        """Test context extraction gets surrounding lines."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        diff = """diff --git a/src/file.tsx b/src/file.tsx
@@ -8,10 +8,10 @@
 line 8
 line 9
 line 10
-<button id="old">
+<button id="new">
 line 12
 line 13
 line 14
"""

        context = analyzer._extract_context_from_diff(diff, 11)

        # Should contain lines around line 11
        assert "line" in context

    def test_handles_empty_diff(self):
        """Test handles empty diff gracefully."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        context = analyzer._extract_context_from_diff("", 10)

        assert context == ""


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestDataclasses:
    """Test dataclass functionality."""

    def test_github_commit_defaults(self):
        """Test GitHubCommit default values."""
        from src.services.github_code_analyzer import GitHubCommit

        commit = GitHubCommit(
            sha="abc123def456",
            short_sha="abc123d",
            author="Test Author",
            author_email="test@example.com",
            date=datetime.now(timezone.utc),
            message="Test commit",
        )

        assert commit.sha == "abc123def456"
        assert commit.files_changed == []  # Default empty list

    def test_github_commit_with_files(self):
        """Test GitHubCommit with files_changed."""
        from src.services.github_code_analyzer import GitHubCommit

        commit = GitHubCommit(
            sha="abc123",
            short_sha="abc123",
            author="Test",
            author_email="test@test.com",
            date=datetime.now(timezone.utc),
            message="Test",
            files_changed=["src/file1.tsx", "src/file2.tsx"],
        )

        assert len(commit.files_changed) == 2

    def test_selector_change_all_fields(self):
        """Test SelectorChange with all fields."""
        from src.services.github_code_analyzer import SelectorChange, GitHubCommit

        commit = GitHubCommit(
            sha="abc123",
            short_sha="abc123",
            author="Test",
            author_email="test@test.com",
            date=datetime.now(timezone.utc),
            message="Test",
        )

        change = SelectorChange(
            old_selector="#old-btn",
            new_selector="[data-testid='new-btn']",
            commit=commit,
            file_path="src/button.tsx",
            line_number=42,
            change_type="modified",
            context="<button>...</button>",
        )

        assert change.old_selector == "#old-btn"
        assert change.new_selector == "[data-testid='new-btn']"
        assert change.commit.sha == "abc123"
        assert change.file_path == "src/button.tsx"
        assert change.line_number == 42
        assert change.change_type == "modified"

    def test_selector_change_none_old_selector(self):
        """Test SelectorChange allows None for old_selector."""
        from src.services.github_code_analyzer import SelectorChange, GitHubCommit

        commit = GitHubCommit(
            sha="abc",
            short_sha="abc",
            author="Test",
            author_email="test@test.com",
            date=datetime.now(timezone.utc),
            message="Test",
        )

        change = SelectorChange(
            old_selector=None,
            new_selector="new-selector",
            commit=commit,
            file_path="file.tsx",
            line_number=1,
            change_type="added",
            context="",
        )

        assert change.old_selector is None
        assert change.change_type == "added"


# ============================================================================
# Client Lifecycle Tests
# ============================================================================


class TestClientLifecycle:
    """Test HTTP client lifecycle."""

    @pytest.mark.asyncio
    async def test_close_clears_client(self):
        """Test close() clears the client."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        # Create a mock client
        mock_client = AsyncMock()
        analyzer._client = mock_client

        await analyzer.close()

        assert analyzer._client is None
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_client(self):
        """Test close() works when no client exists."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        # Should not raise
        await analyzer.close()

        assert analyzer._client is None


# ============================================================================
# API Request Error Handling Tests
# ============================================================================


class TestAPIErrorHandling:
    """Test API error handling."""

    @pytest.mark.asyncio
    async def test_404_returns_none(self):
        """Test 404 response returns None."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        analyzer._client = mock_client

        result = await analyzer._api_request("/repos/org/repo/contents/nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self):
        """Test HTTP error returns None."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer
        import httpx

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        analyzer._client = mock_client

        result = await analyzer._api_request("/repos/org/repo/commits")

        assert result is None


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestGetGithubAnalyzerForProject:
    """Test the factory function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_supabase_not_configured(self):
        """Test returns None when Supabase is not configured."""
        from src.services.github_code_analyzer import get_github_analyzer_for_project

        mock_supabase = MagicMock()
        mock_supabase.is_configured = False

        with patch(
            "src.services.supabase_client.get_supabase_client",
            return_value=mock_supabase,
        ):
            result = await get_github_analyzer_for_project("proj_id", "org_id")

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_repo_url(self):
        """Test returns None when project has no repository_url."""
        from src.services.github_code_analyzer import get_github_analyzer_for_project

        mock_supabase = MagicMock()
        mock_supabase.is_configured = True
        mock_supabase.request = AsyncMock(
            return_value={"data": [{"repository_url": None}]}
        )

        with patch(
            "src.services.supabase_client.get_supabase_client",
            return_value=mock_supabase,
        ):
            result = await get_github_analyzer_for_project("proj_id", "org_id")

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_github_integration(self):
        """Test returns None when no GitHub integration exists."""
        from src.services.github_code_analyzer import get_github_analyzer_for_project

        mock_supabase = MagicMock()
        mock_supabase.is_configured = True

        async def mock_request(endpoint):
            if "projects" in endpoint:
                return {"data": [{"repository_url": "https://github.com/org/repo"}]}
            elif "integrations" in endpoint:
                return {"data": []}  # No integrations
            return {"data": []}

        mock_supabase.request = mock_request

        with patch(
            "src.services.supabase_client.get_supabase_client",
            return_value=mock_supabase,
        ):
            result = await get_github_analyzer_for_project("proj_id", "org_id")

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_analyzer_when_configured(self):
        """Test returns analyzer when properly configured."""
        from src.services.github_code_analyzer import get_github_analyzer_for_project

        mock_supabase = MagicMock()
        mock_supabase.is_configured = True

        async def mock_request(endpoint):
            if "projects" in endpoint:
                return {
                    "data": [
                        {"repository_url": "https://github.com/testorg/testrepo"}
                    ]
                }
            elif "integrations" in endpoint:
                return {"data": [{"credentials": {"access_token": "ghp_testtoken"}}]}
            return {"data": []}

        mock_supabase.request = mock_request

        with patch(
            "src.services.supabase_client.get_supabase_client",
            return_value=mock_supabase,
        ):
            result = await get_github_analyzer_for_project("proj_id", "org_id")

            assert result is not None
            assert result.owner == "testorg"
            assert result.repo == "testrepo"


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_get_recent_commits_empty_response(self):
        """Test handling of empty commits response."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        analyzer._client = mock_client

        commits = await analyzer.get_recent_commits()

        assert commits == []

    @pytest.mark.asyncio
    async def test_get_file_content_base64_decoding(self):
        """Test file content base64 decoding."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer
        import base64

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        test_content = "Hello, World!"
        encoded_content = base64.b64encode(test_content.encode()).decode()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": encoded_content}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        analyzer._client = mock_client

        content = await analyzer.get_file_content("src/test.py")

        assert content == test_content

    def test_headers_include_auth(self):
        """Test headers include authorization."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer(
            "https://github.com/org/repo", "test_token_123"
        )
        headers = analyzer._headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token_123"
        assert "User-Agent" in headers
        assert "Argus" in headers["User-Agent"]

    @pytest.mark.asyncio
    async def test_find_selector_changes_returns_list(self):
        """Test find_selector_changes returns a list."""
        from src.services.github_code_analyzer import GitHubCodeAnalyzer

        analyzer = GitHubCodeAnalyzer("https://github.com/org/repo", "token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        analyzer._client = mock_client

        changes = await analyzer.find_selector_changes("some-selector")

        assert isinstance(changes, list)
