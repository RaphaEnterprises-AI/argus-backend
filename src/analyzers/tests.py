"""Test Analyzer - pytest, jest, Playwright test pattern analysis.

Understands:
- Test structure and organization
- Test fixtures and mocks
- Assertion patterns
- Test coverage indicators
- E2E test selectors
"""

import logging
import re

from src.indexer import ParsedFile

from .base import (
    BaseAnalyzer,
    ComponentInfo,
    ComponentType,
    Issue,
    Severity,
)

logger = logging.getLogger(__name__)


class TestAnalyzer(BaseAnalyzer):
    """Analyzer for test files (pytest, jest, Playwright, Cypress)."""

    @property
    def analyzer_type(self) -> str:
        return "tests"

    def get_file_patterns(self) -> list[str]:
        return [
            # Python tests
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/**/*.py",
            "**/conftest.py",
            # JavaScript/TypeScript tests
            "**/*.test.ts",
            "**/*.test.tsx",
            "**/*.test.js",
            "**/*.test.jsx",
            "**/*.spec.ts",
            "**/*.spec.tsx",
            "**/*.spec.js",
            # E2E tests
            "**/e2e/**/*.ts",
            "**/e2e/**/*.js",
            "**/playwright/**/*.ts",
            "**/cypress/**/*.ts",
            "**/cypress/**/*.js",
        ]

    def analyze_file(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze a test file."""
        components: list[ComponentInfo] = []

        framework = self._detect_test_framework(parsed)

        if framework == "pytest":
            components.extend(self._analyze_pytest(parsed))
        elif framework == "jest":
            components.extend(self._analyze_jest(parsed))
        elif framework == "playwright":
            components.extend(self._analyze_playwright(parsed))
        elif framework == "cypress":
            components.extend(self._analyze_cypress(parsed))
        elif framework == "conftest":
            components.extend(self._analyze_conftest(parsed))

        return components

    def _detect_test_framework(self, parsed: ParsedFile) -> str:
        """Detect which test framework is being used."""
        content = parsed.content
        file_path = parsed.file_path.lower()

        # Conftest is special
        if file_path.endswith("conftest.py"):
            return "conftest"

        # E2E frameworks
        if "@playwright" in content or "from playwright" in content:
            return "playwright"
        if "cypress" in file_path or "cy." in content:
            return "cypress"

        # Unit test frameworks
        if file_path.endswith(".py"):
            return "pytest"
        if any(file_path.endswith(ext) for ext in [".test.ts", ".test.tsx", ".test.js", ".spec.ts"]):
            return "jest"

        return "unknown"

    def _analyze_pytest(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze pytest test files."""
        components: list[ComponentInfo] = []

        # Find test functions
        for func in parsed.get_functions():
            name = func.name
            if not name:
                continue

            # Check if it's a test function
            if name.startswith("test_"):
                component = ComponentInfo(
                    name=name,
                    component_type=ComponentType.TEST_CASE,
                    file_path=parsed.file_path,
                    start_line=func.start_line,
                    end_line=func.end_line,
                )

                # Analyze test quality
                component.issues.extend(self._check_pytest_issues(func, name))

                # Extract assertions
                assertions = self._count_assertions(func.text, "pytest")
                component.methods = [f"assertions: {assertions}"]

                components.append(component)

        # Find test classes
        for cls in parsed.get_classes():
            name = cls.name
            if not name:
                continue

            if name.startswith("Test"):
                # Count test methods
                test_methods = []
                for method in cls.find_descendants("function_definition"):
                    method_name = method.name
                    if method_name and method_name.startswith("test_"):
                        test_methods.append(method_name)

                component = ComponentInfo(
                    name=name,
                    component_type=ComponentType.TEST_SUITE,
                    file_path=parsed.file_path,
                    start_line=cls.start_line,
                    end_line=cls.end_line,
                    methods=test_methods,
                )

                if len(test_methods) == 0:
                    component.issues.append(Issue(
                        severity=Severity.WARNING,
                        message=f"Test class {name} has no test methods",
                        file_path=parsed.file_path,
                        line_number=cls.start_line,
                        code="TEST_EMPTY_CLASS",
                    ))

                components.append(component)

        return components

    def _check_pytest_issues(self, func, name: str) -> list[Issue]:
        """Check pytest test function for issues."""
        issues = []
        func_text = func.text

        # 1. No assertions
        assertion_count = self._count_assertions(func_text, "pytest")
        if assertion_count == 0:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Test {name} has no assertions",
                file_path="",
                line_number=func.start_line,
                code="TEST_NO_ASSERTIONS",
            ))

        # 2. Too many assertions (might need splitting)
        if assertion_count > 10:
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"Test {name} has {assertion_count} assertions - consider splitting",
                file_path="",
                line_number=func.start_line,
                code="TEST_TOO_MANY_ASSERTIONS",
            ))

        # 3. Missing docstring
        if '"""' not in func_text[:200] and "'''" not in func_text[:200]:
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"Test {name} lacks docstring",
                file_path="",
                line_number=func.start_line,
                code="TEST_NO_DOCSTRING",
            ))

        # 4. Sleep in test (flaky)
        if "time.sleep" in func_text or "asyncio.sleep" in func_text:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Test {name} uses sleep - may be flaky",
                file_path="",
                line_number=func.start_line,
                suggestion="Use explicit waits or mock time",
                code="TEST_USES_SLEEP",
            ))

        # 5. Bare except
        if "except:" in func_text and "except Exception" not in func_text:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Test {name} uses bare except",
                file_path="",
                line_number=func.start_line,
                code="TEST_BARE_EXCEPT",
            ))

        return issues

    def _count_assertions(self, text: str, framework: str) -> int:
        """Count assertions in test code."""
        if framework == "pytest":
            # Python assert statements
            return len(re.findall(r'\bassert\b', text))
        elif framework == "jest":
            # Jest expects
            return len(re.findall(r'\bexpect\s*\(', text))
        elif framework in ["playwright", "cypress"]:
            # E2E assertions
            return len(re.findall(r'\b(?:expect|should|assert|toBe|toEqual)\s*\(', text))
        return 0

    def _analyze_jest(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Jest test files."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find describe blocks (test suites)
        describe_pattern = r'describe\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(describe_pattern, content):
            suite_name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            # Find the end of this describe block
            block_start = content.find("{", match.end())
            if block_start == -1:
                continue

            brace_count = 1
            block_end = block_start + 1

            for i, char in enumerate(content[block_start + 1:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        block_end = block_start + 1 + i
                        break

            block_content = content[block_start:block_end]

            # Find test cases within this block
            test_pattern = r'(?:it|test)\s*\(\s*["\']([^"\']+)["\']'
            test_methods = []

            for test_match in re.finditer(test_pattern, block_content):
                test_methods.append(test_match.group(1))

            component = ComponentInfo(
                name=suite_name,
                component_type=ComponentType.TEST_SUITE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + block_content.count("\n"),
                methods=test_methods,
            )

            # Check for issues
            component.issues.extend(self._check_jest_issues(block_content, suite_name))

            components.append(component)

        # Find standalone test cases
        standalone_pattern = r'^(?:it|test)\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(standalone_pattern, content, re.MULTILINE):
            test_name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            # Get the test block
            block_start = content.find("{", match.end())
            if block_start == -1:
                continue

            brace_count = 1
            block_end = block_start + 1

            for i, char in enumerate(content[block_start + 1:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        block_end = block_start + 1 + i
                        break

            block_content = content[block_start:block_end]

            component = ComponentInfo(
                name=test_name,
                component_type=ComponentType.TEST_CASE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + block_content.count("\n"),
            )

            assertion_count = self._count_assertions(block_content, "jest")
            component.methods = [f"assertions: {assertion_count}"]

            if assertion_count == 0:
                component.issues.append(Issue(
                    severity=Severity.WARNING,
                    message=f"Test '{test_name}' has no assertions",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    code="TEST_NO_ASSERTIONS",
                ))

            components.append(component)

        return components

    def _check_jest_issues(self, content: str, suite_name: str) -> list[Issue]:
        """Check Jest test suite for issues."""
        issues = []

        # 1. Using setTimeout directly
        if "setTimeout" in content and "jest.useFakeTimers" not in content:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Suite '{suite_name}' uses setTimeout - may be flaky",
                file_path="",
                suggestion="Use jest.useFakeTimers() or waitFor()",
                code="TEST_USES_TIMEOUT",
            ))

        # 2. Skipped tests
        if ".skip" in content or "xit(" in content or "xdescribe(" in content:
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"Suite '{suite_name}' has skipped tests",
                file_path="",
                code="TEST_SKIPPED",
            ))

        # 3. Only focused tests
        if ".only" in content:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Suite '{suite_name}' has .only - other tests won't run",
                file_path="",
                code="TEST_ONLY_FOCUS",
            ))

        # 4. No cleanup (afterEach/afterAll)
        if "beforeEach" in content and "afterEach" not in content:
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"Suite '{suite_name}' has beforeEach but no afterEach cleanup",
                file_path="",
                code="TEST_NO_CLEANUP",
            ))

        return issues

    def _analyze_playwright(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Playwright test files."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find test.describe blocks
        describe_pattern = r'test\.describe\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(describe_pattern, content):
            suite_name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            component = ComponentInfo(
                name=suite_name,
                component_type=ComponentType.TEST_SUITE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 20,  # Approximate
            )

            components.append(component)

        # Find individual tests
        test_pattern = r'test\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(test_pattern, content):
            # Skip if this is test.describe
            if content[match.start() - 1:match.start()] == '.':
                continue

            test_name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            # Get test block content
            block_start = content.find("{", match.end())
            if block_start == -1:
                continue

            brace_count = 1
            block_end = block_start + 1

            for i, char in enumerate(content[block_start + 1:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        block_end = block_start + 1 + i
                        break

            block_content = content[block_start:block_end]

            component = ComponentInfo(
                name=test_name,
                component_type=ComponentType.TEST_CASE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + block_content.count("\n"),
            )

            # Extract selectors used in test
            component.selectors = self._extract_playwright_selectors(block_content)

            # Check for issues
            component.issues.extend(self._check_playwright_issues(block_content, test_name))

            components.append(component)

        return components

    def _extract_playwright_selectors(self, content: str) -> list[str]:
        """Extract Playwright selectors from test content."""
        selectors = []

        # page.locator patterns
        locator_patterns = [
            r'locator\s*\(\s*["\']([^"\']+)["\']',
            r'getByRole\s*\(\s*["\']([^"\']+)["\']',
            r'getByText\s*\(\s*["\']([^"\']+)["\']',
            r'getByTestId\s*\(\s*["\']([^"\']+)["\']',
            r'getByLabel\s*\(\s*["\']([^"\']+)["\']',
            r'getByPlaceholder\s*\(\s*["\']([^"\']+)["\']',
        ]

        for pattern in locator_patterns:
            for match in re.finditer(pattern, content):
                selectors.append(match.group(1))

        return selectors

    def _check_playwright_issues(self, content: str, test_name: str) -> list[Issue]:
        """Check Playwright test for issues."""
        issues = []

        # 1. Hard-coded waits
        if "page.waitForTimeout" in content:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Test '{test_name}' uses waitForTimeout - prefer explicit waits",
                file_path="",
                suggestion="Use waitForSelector, waitForLoadState, or expect with timeout",
                code="E2E_HARDCODED_WAIT",
            ))

        # 2. No assertions
        assertion_count = self._count_assertions(content, "playwright")
        if assertion_count == 0:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Test '{test_name}' has no assertions",
                file_path="",
                code="TEST_NO_ASSERTIONS",
            ))

        # 3. Fragile CSS selectors
        fragile_patterns = [
            r'locator\s*\(\s*["\'][.#]?\w+\s*>\s*\w+\s*>\s*\w+',  # Deep nesting
            r'locator\s*\(\s*["\']\.[\w-]+\s+\.[\w-]+\s+\.[\w-]+',  # Multiple classes
            r':nth-child\(',  # Positional
        ]

        for pattern in fragile_patterns:
            if re.search(pattern, content):
                issues.append(Issue(
                    severity=Severity.INFO,
                    message=f"Test '{test_name}' uses fragile selectors",
                    file_path="",
                    suggestion="Prefer data-testid or getByRole for stability",
                    code="E2E_FRAGILE_SELECTOR",
                ))
                break

        # 4. Missing test isolation
        if "page.goto" not in content and "beforeEach" not in content:
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"Test '{test_name}' may not navigate - check test isolation",
                file_path="",
                code="E2E_NO_NAVIGATION",
            ))

        return issues

    def _analyze_cypress(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Cypress test files."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find describe blocks
        describe_pattern = r'describe\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(describe_pattern, content):
            suite_name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            component = ComponentInfo(
                name=suite_name,
                component_type=ComponentType.TEST_SUITE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 20,
            )

            components.append(component)

        # Find individual tests
        test_pattern = r'it\s*\(\s*["\']([^"\']+)["\']'

        for match in re.finditer(test_pattern, content):
            test_name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            block_start = content.find("{", match.end())
            if block_start == -1:
                continue

            brace_count = 1
            block_end = block_start + 1

            for i, char in enumerate(content[block_start + 1:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        block_end = block_start + 1 + i
                        break

            block_content = content[block_start:block_end]

            component = ComponentInfo(
                name=test_name,
                component_type=ComponentType.TEST_CASE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + block_content.count("\n"),
            )

            # Extract selectors
            component.selectors = self._extract_cypress_selectors(block_content)

            # Check for issues
            component.issues.extend(self._check_cypress_issues(block_content, test_name))

            components.append(component)

        return components

    def _extract_cypress_selectors(self, content: str) -> list[str]:
        """Extract Cypress selectors from test content."""
        selectors = []

        selector_patterns = [
            r'cy\.get\s*\(\s*["\']([^"\']+)["\']',
            r'cy\.contains\s*\(\s*["\']([^"\']+)["\']',
            r'cy\.findByTestId\s*\(\s*["\']([^"\']+)["\']',
            r'cy\.findByRole\s*\(\s*["\']([^"\']+)["\']',
        ]

        for pattern in selector_patterns:
            for match in re.finditer(pattern, content):
                selectors.append(match.group(1))

        return selectors

    def _check_cypress_issues(self, content: str, test_name: str) -> list[Issue]:
        """Check Cypress test for issues."""
        issues = []

        # 1. cy.wait with number (anti-pattern)
        if re.search(r'cy\.wait\s*\(\s*\d+', content):
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Test '{test_name}' uses cy.wait(ms) - prefer aliased routes",
                file_path="",
                suggestion="Use cy.wait('@alias') for network requests",
                code="E2E_HARDCODED_WAIT",
            ))

        # 2. No assertions
        if ".should(" not in content and ".expect(" not in content:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Test '{test_name}' may lack assertions",
                file_path="",
                code="TEST_NO_ASSERTIONS",
            ))

        # 3. Mixing async/await with Cypress commands
        if "async " in content or "await " in content:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Test '{test_name}' uses async/await - not recommended with Cypress",
                file_path="",
                suggestion="Use Cypress's built-in command chaining",
                code="CYPRESS_ASYNC_ANTIPATTERN",
            ))

        return issues

    def _analyze_conftest(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze pytest conftest.py files."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find fixtures
        fixture_pattern = r'@pytest\.fixture(?:\([^)]*\))?\s*\n(?:async\s+)?def\s+(\w+)'

        for match in re.finditer(fixture_pattern, content):
            fixture_name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            component = ComponentInfo(
                name=fixture_name,
                component_type=ComponentType.FIXTURE,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + 10,
            )

            # Check scope
            scope_match = re.search(r'scope\s*=\s*["\'](\w+)["\']', match.group(0))
            if scope_match:
                component.methods = [f"scope: {scope_match.group(1)}"]

            components.append(component)

        # Find hooks
        hook_names = ["pytest_configure", "pytest_collection_modifyitems", "pytest_runtest_setup"]
        for hook_name in hook_names:
            if f"def {hook_name}" in content:
                line_number = content.find(f"def {hook_name}")
                line_number = content[:line_number].count("\n") + 1

                component = ComponentInfo(
                    name=hook_name,
                    component_type=ComponentType.FIXTURE,
                    file_path=parsed.file_path,
                    start_line=line_number,
                    end_line=line_number + 10,
                )
                components.append(component)

        return components
