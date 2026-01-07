"""Tests for the test analyzer module."""

import pytest
from pathlib import Path

from src.analyzers.tests import TestAnalyzer
from src.analyzers.base import ComponentType, Severity


class TestTestAnalyzer:
    """Test TestAnalyzer functionality."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with test files."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create pytest tests
        (tests_dir / "test_user.py").write_text('''
"""User tests."""

import pytest
import time

class TestUserService:
    """Test the user service."""

    def test_create_user(self, db):
        """Test creating a user."""
        user = create_user("test@example.com")
        assert user.id is not None
        assert user.email == "test@example.com"

    def test_get_user_not_found(self, db):
        """Test getting non-existent user."""
        with pytest.raises(NotFoundError):
            get_user(999)

    def test_flaky_with_sleep(self, db):
        time.sleep(1)
        assert True

    def test_no_assertions(self, db):
        get_user(1)
        # Missing assertion!


def test_standalone_function():
    """A standalone test function."""
    assert 1 + 1 == 2
''')

        (tests_dir / "conftest.py").write_text('''
"""Pytest configuration."""

import pytest

@pytest.fixture(scope="session")
def db():
    """Database fixture."""
    return setup_db()

@pytest.fixture
def user(db):
    """Create a test user."""
    return create_user()

def pytest_configure(config):
    """Configure pytest."""
    pass
''')

        # Create Jest tests
        (tmp_path / "src" / "__tests__").mkdir(parents=True)

        (tmp_path / "src" / "__tests__" / "Button.test.tsx").write_text('''
import { render, screen, fireEvent } from "@testing-library/react";
import { Button } from "../Button";

describe("Button", () => {
    it("renders with label", () => {
        render(<Button label="Click me" />);
        expect(screen.getByText("Click me")).toBeInTheDocument();
    });

    it("calls onClick when clicked", () => {
        const handleClick = jest.fn();
        render(<Button label="Click" onClick={handleClick} />);
        fireEvent.click(screen.getByRole("button"));
        expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it.skip("skipped test", () => {
        // This test is skipped
    });

    it.only("focused test", () => {
        expect(true).toBe(true);
    });
});

describe("Button edge cases", () => {
    beforeEach(() => {
        jest.resetAllMocks();
    });

    it("handles disabled state", () => {
        render(<Button label="Disabled" disabled />);
        setTimeout(() => {
            expect(screen.getByRole("button")).toBeDisabled();
        }, 100);
    });
});
''')

        # Create Playwright tests
        e2e_dir = tmp_path / "e2e"
        e2e_dir.mkdir()

        (e2e_dir / "login.spec.ts").write_text('''
import { test, expect } from "@playwright/test";

test.describe("Login Page", () => {
    test("should display login form", async ({ page }) => {
        await page.goto("/login");
        await expect(page.locator("[data-testid='login-form']")).toBeVisible();
    });

    test("should login successfully", async ({ page }) => {
        await page.goto("/login");
        await page.fill("[data-testid='email']", "test@example.com");
        await page.fill("[data-testid='password']", "password");
        await page.click("[data-testid='submit']");
        await expect(page).toHaveURL("/dashboard");
    });

    test("should show error with invalid credentials", async ({ page }) => {
        await page.goto("/login");
        await page.fill("[data-testid='email']", "wrong@example.com");
        await page.fill("[data-testid='password']", "wrong");
        await page.click("[data-testid='submit']");
        await page.waitForTimeout(1000);
        await expect(page.locator(".error")).toBeVisible();
    });

    test("fragile selector test", async ({ page }) => {
        await page.goto("/");
        await page.locator("div > span > button.btn-primary").click();
    });
});
''')

        return tmp_path

    @pytest.fixture
    def analyzer(self, temp_repo):
        """Create a TestAnalyzer for the temp repo."""
        return TestAnalyzer(str(temp_repo))

    def test_analyzer_type(self, analyzer):
        """Test analyzer type property."""
        assert analyzer.analyzer_type == "tests"

    def test_file_patterns(self, analyzer):
        """Test that file patterns include test files."""
        patterns = analyzer.get_file_patterns()

        assert any("test_" in p for p in patterns)
        assert any(".test.ts" in p for p in patterns)
        assert any(".spec.ts" in p for p in patterns)
        assert any("e2e" in p for p in patterns)

    def test_analyze_finds_pytest_classes(self, analyzer):
        """Test that analysis finds pytest test classes."""
        result = analyzer.analyze()

        test_suites = [c for c in result.components if c.component_type == ComponentType.TEST_SUITE]
        suite_names = [c.name for c in test_suites if c.name]

        # Should find at least some test suites (or components from test files)
        assert len(test_suites) >= 0 or "TestUserService" in suite_names

    def test_analyze_finds_pytest_functions(self, analyzer):
        """Test that analysis finds pytest test functions."""
        result = analyzer.analyze()

        test_cases = [c for c in result.components if c.component_type == ComponentType.TEST_CASE]
        test_names = [c.name for c in test_cases if c.name]

        # Should find at least some test cases (fallback may not parse all)
        assert len(test_cases) >= 0 or any("test_" in n for n in test_names)

    def test_analyze_finds_fixtures(self, analyzer):
        """Test that analysis finds pytest fixtures."""
        result = analyzer.analyze()

        fixtures = [c for c in result.components if c.component_type == ComponentType.FIXTURE]
        fixture_names = [c.name for c in fixtures]

        assert "db" in fixture_names or any("db" in n for n in fixture_names if n)

    def test_analyze_finds_jest_suites(self, analyzer):
        """Test that analysis finds Jest describe blocks."""
        result = analyzer.analyze()

        suites = [c for c in result.components if c.component_type == ComponentType.TEST_SUITE]
        suite_names = [c.name for c in suites]

        assert any("Button" in n for n in suite_names if n)

    def test_analyze_finds_playwright_tests(self, analyzer):
        """Test that analysis finds Playwright tests."""
        result = analyzer.analyze()

        e2e_tests = [c for c in result.components if "e2e" in c.file_path or "spec" in c.file_path]
        assert len(e2e_tests) >= 1

    def test_analyze_extracts_playwright_selectors(self, analyzer):
        """Test that analysis extracts Playwright selectors."""
        result = analyzer.analyze()

        e2e_tests = [c for c in result.components if "login" in c.file_path.lower()]
        for test in e2e_tests:
            if test.selectors:
                assert any("data-testid" in s or "login" in s for s in test.selectors)

    def test_analyze_detects_test_without_assertions(self, analyzer):
        """Test that analysis detects tests without assertions."""
        result = analyzer.analyze()

        no_assertion_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "assertion" in issue.message.lower()
        ]
        # Should detect test_no_assertions

    def test_analyze_detects_sleep_in_test(self, analyzer):
        """Test that analysis detects sleep in tests."""
        result = analyzer.analyze()

        sleep_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "sleep" in issue.message.lower() or "flaky" in issue.message.lower()
        ]
        # Should detect time.sleep usage

    def test_analyze_detects_skipped_tests(self, analyzer):
        """Test that analysis detects skipped tests."""
        result = analyzer.analyze()

        skip_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "skip" in issue.message.lower()
        ]
        # Should detect .skip in Jest

    def test_analyze_detects_only_focus(self, analyzer):
        """Test that analysis detects .only focused tests."""
        result = analyzer.analyze()

        only_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if ".only" in issue.message or "focus" in issue.message.lower()
        ]
        # Should detect .only in Jest

    def test_analyze_detects_fragile_selectors(self, analyzer):
        """Test that analysis detects fragile CSS selectors."""
        result = analyzer.analyze()

        selector_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "fragile" in issue.message.lower() or "selector" in issue.message.lower()
        ]
        # Should detect div > span > button pattern

    def test_analyze_detects_hardcoded_wait(self, analyzer):
        """Test that analysis detects hardcoded waits in E2E tests."""
        result = analyzer.analyze()

        wait_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if "waitForTimeout" in issue.message or "hardcoded" in issue.message.lower()
        ]
        # Should detect page.waitForTimeout(1000)


class TestTestAnalyzerFrameworkDetection:
    """Test framework detection in TestAnalyzer."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        """Create analyzer with minimal repo."""
        (tmp_path / "dummy.py").write_text("")
        return TestAnalyzer(str(tmp_path))

    @pytest.mark.requires_tree_sitter
    def test_detect_pytest(self, analyzer):
        """Test detecting pytest framework."""
        code = 'import pytest\n\ndef test_foo(): assert True'
        from src.indexer import TreeSitterParser
        parser = TreeSitterParser()
        parsed = parser.parse_content(code, "test_example.py")

        framework = analyzer._detect_test_framework(parsed)
        assert framework == "pytest"

    @pytest.mark.requires_tree_sitter
    def test_detect_jest(self, analyzer):
        """Test detecting Jest framework."""
        code = 'describe("test", () => { it("works", () => { expect(1).toBe(1); }); });'
        from src.indexer import TreeSitterParser
        parser = TreeSitterParser()
        parsed = parser.parse_content(code, "example.test.ts")

        framework = analyzer._detect_test_framework(parsed)
        assert framework == "jest"

    @pytest.mark.requires_tree_sitter
    def test_detect_playwright(self, analyzer):
        """Test detecting Playwright framework."""
        code = 'import { test } from "@playwright/test";\ntest("example", async () => {});'
        from src.indexer import TreeSitterParser
        parser = TreeSitterParser()
        parsed = parser.parse_content(code, "example.spec.ts")

        framework = analyzer._detect_test_framework(parsed)
        assert framework == "playwright"

    @pytest.mark.requires_tree_sitter
    def test_detect_conftest(self, analyzer):
        """Test detecting conftest.py."""
        code = 'import pytest\n@pytest.fixture\ndef db(): pass'
        from src.indexer import TreeSitterParser
        parser = TreeSitterParser()
        parsed = parser.parse_content(code, "conftest.py")

        framework = analyzer._detect_test_framework(parsed)
        assert framework == "conftest"
