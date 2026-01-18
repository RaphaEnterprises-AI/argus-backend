"""Base Analyzer - Foundation for specialized code analyzers.

Provides common functionality for all specialized analyzers:
- File discovery
- Pattern matching
- Result aggregation
- Caching
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.indexer import ASTNode, ParsedFile, TreeSitterParser

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Severity levels for issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Types of components/entities."""
    # Frontend
    COMPONENT = "component"
    HOOK = "hook"
    PAGE = "page"
    LAYOUT = "layout"
    CONTEXT = "context"
    STORE = "store"

    # Backend
    ROUTE = "route"
    CONTROLLER = "controller"
    SERVICE = "service"
    MIDDLEWARE = "middleware"
    MODEL = "model"

    # Database
    TABLE = "table"
    QUERY = "query"
    MIGRATION = "migration"
    INDEX = "index"
    TRIGGER = "trigger"

    # Microservices
    DEPLOYMENT = "deployment"
    SERVICE_MESH = "service"
    CONFIG_MAP = "configmap"
    SECRET = "secret"
    INGRESS = "ingress"

    # Tests
    TEST_SUITE = "test_suite"
    TEST_CASE = "test_case"
    FIXTURE = "fixture"
    MOCK = "mock"


@dataclass
class Issue:
    """An issue or recommendation found during analysis."""
    severity: Severity
    message: str
    file_path: str
    line_number: int | None = None
    suggestion: str | None = None
    code: str | None = None  # Issue code for categorization


@dataclass
class ComponentInfo:
    """Information about a component/entity."""
    name: str
    component_type: ComponentType
    file_path: str
    start_line: int
    end_line: int

    # Relationships
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    used_by: list[str] = field(default_factory=list)

    # Metadata
    props: list[str] = field(default_factory=list)  # For React components
    methods: list[str] = field(default_factory=list)
    selectors: list[str] = field(default_factory=list)  # data-testid, etc.

    # Quality
    has_tests: bool = False
    test_file: str | None = None
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "file_path": self.file_path,
            "lines": f"{self.start_line}-{self.end_line}",
            "imports": self.imports,
            "exports": self.exports,
            "dependencies": self.dependencies,
            "selectors": self.selectors,
            "has_tests": self.has_tests,
            "issues": [
                {"severity": i.severity.value, "message": i.message}
                for i in self.issues
            ],
        }


@dataclass
class RouteInfo:
    """Information about an API route/endpoint."""
    path: str
    method: str  # GET, POST, PUT, DELETE, etc.
    handler_name: str
    file_path: str
    line_number: int

    # Parameters
    path_params: list[str] = field(default_factory=list)
    query_params: list[str] = field(default_factory=list)
    body_schema: str | None = None

    # Response
    response_schema: str | None = None
    status_codes: list[int] = field(default_factory=list)

    # Security
    requires_auth: bool = False
    permissions: list[str] = field(default_factory=list)

    # Quality
    has_tests: bool = False
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "handler": self.handler_name,
            "file": self.file_path,
            "line": self.line_number,
            "path_params": self.path_params,
            "query_params": self.query_params,
            "requires_auth": self.requires_auth,
            "has_tests": self.has_tests,
        }


@dataclass
class QueryInfo:
    """Information about a database query."""
    query_type: str  # SELECT, INSERT, UPDATE, DELETE, CREATE, etc.
    tables: list[str]
    file_path: str
    line_number: int

    # Query details
    raw_query: str | None = None
    columns: list[str] = field(default_factory=list)
    joins: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)

    # Performance
    has_index: bool = False
    estimated_complexity: str = "unknown"

    # Security
    uses_parameterization: bool = True
    sql_injection_risk: bool = False

    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.query_type,
            "tables": self.tables,
            "file": self.file_path,
            "line": self.line_number,
            "parameterized": self.uses_parameterization,
            "sql_injection_risk": self.sql_injection_risk,
        }


@dataclass
class AnalysisResult:
    """Result of analyzing a repository."""
    repo_path: str
    analyzer_type: str
    files_analyzed: int
    analysis_time_ms: float

    # Discovered entities
    components: list[ComponentInfo] = field(default_factory=list)
    routes: list[RouteInfo] = field(default_factory=list)
    queries: list[QueryInfo] = field(default_factory=list)

    # Quality metrics
    total_issues: int = 0
    issues_by_severity: dict = field(default_factory=dict)

    # Coverage
    components_with_tests: int = 0
    routes_with_tests: int = 0
    test_coverage_estimate: float = 0.0

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "repo_path": self.repo_path,
            "analyzer": self.analyzer_type,
            "files_analyzed": self.files_analyzed,
            "analysis_time_ms": self.analysis_time_ms,
            "summary": {
                "components": len(self.components),
                "routes": len(self.routes),
                "queries": len(self.queries),
                "total_issues": self.total_issues,
                "test_coverage": f"{self.test_coverage_estimate:.1%}",
            },
            "issues_by_severity": self.issues_by_severity,
            "recommendations": self.recommendations,
        }


class BaseAnalyzer(ABC):
    """Base class for specialized code analyzers.

    Subclasses should implement:
    - get_file_patterns(): Which files to analyze
    - analyze_file(): How to analyze a single file
    - aggregate_results(): How to combine results
    """

    def __init__(self, repo_path: str):
        """Initialize analyzer.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path).resolve()
        self.parser = TreeSitterParser()
        self._cache: dict[str, ParsedFile] = {}

    @property
    @abstractmethod
    def analyzer_type(self) -> str:
        """Return the type of analyzer (e.g., 'frontend', 'backend')."""
        pass

    @abstractmethod
    def get_file_patterns(self) -> list[str]:
        """Return glob patterns for files to analyze.

        Returns:
            List of glob patterns (e.g., ["**/*.tsx", "**/*.jsx"])
        """
        pass

    @abstractmethod
    def analyze_file(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze a single parsed file.

        Args:
            parsed: ParsedFile from tree-sitter

        Returns:
            List of ComponentInfo found in the file
        """
        pass

    def analyze(self) -> AnalysisResult:
        """Analyze the entire repository.

        Returns:
            AnalysisResult with all findings
        """
        import time
        start = time.perf_counter()

        all_components: list[ComponentInfo] = []
        all_routes: list[RouteInfo] = []
        all_queries: list[QueryInfo] = []
        files_analyzed = 0

        # Find and analyze files
        for pattern in self.get_file_patterns():
            for file_path in self.repo_path.glob(pattern):
                if self._should_skip(file_path):
                    continue

                try:
                    parsed = self._parse_file(file_path)
                    if parsed:
                        components = self.analyze_file(parsed)
                        all_components.extend(components)
                        files_analyzed += 1

                except Exception as e:
                    logger.debug(f"Error analyzing {file_path}: {e}")

        # Aggregate and analyze relationships
        self._analyze_relationships(all_components)
        self._find_tests(all_components)

        # Calculate metrics
        elapsed_ms = (time.perf_counter() - start) * 1000
        issues_by_severity = self._count_issues(all_components)
        total_issues = sum(issues_by_severity.values())

        components_with_tests = sum(1 for c in all_components if c.has_tests)
        test_coverage = components_with_tests / max(len(all_components), 1)

        return AnalysisResult(
            repo_path=str(self.repo_path),
            analyzer_type=self.analyzer_type,
            files_analyzed=files_analyzed,
            analysis_time_ms=elapsed_ms,
            components=all_components,
            routes=all_routes,
            queries=all_queries,
            total_issues=total_issues,
            issues_by_severity=issues_by_severity,
            components_with_tests=components_with_tests,
            test_coverage_estimate=test_coverage,
            recommendations=self._generate_recommendations(all_components),
        )

    def _parse_file(self, file_path: Path) -> ParsedFile | None:
        """Parse a file with caching."""
        key = str(file_path)
        if key not in self._cache:
            self._cache[key] = self.parser.parse_file(str(file_path))
        return self._cache[key]

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            "node_modules",
            "__pycache__",
            ".git",
            "dist",
            "build",
            ".next",
            "venv",
            ".venv",
        ]
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)

    def _analyze_relationships(self, components: list[ComponentInfo]) -> None:
        """Analyze relationships between components."""
        component_names = {c.name for c in components}

        for component in components:
            # Find dependencies based on imports
            for imp in component.imports:
                # Extract module/component name from import
                name = imp.split("/")[-1].split(".")[0]
                if name in component_names and name != component.name:
                    component.dependencies.append(name)

    def _find_tests(self, components: list[ComponentInfo]) -> None:
        """Find test files for components."""
        test_patterns = [
            "**/*.test.ts",
            "**/*.test.tsx",
            "**/*.test.js",
            "**/*.spec.ts",
            "**/*.spec.tsx",
            "**/test_*.py",
            "**/*_test.py",
            "**/*_test.go",
        ]

        test_files: set[str] = set()
        for pattern in test_patterns:
            for f in self.repo_path.glob(pattern):
                test_files.add(str(f))

        for component in components:
            # Check for matching test file
            base_name = Path(component.file_path).stem
            for test_file in test_files:
                test_base = Path(test_file).stem
                if base_name in test_base or component.name.lower() in test_base.lower():
                    component.has_tests = True
                    component.test_file = test_file
                    break

    def _count_issues(self, components: list[ComponentInfo]) -> dict:
        """Count issues by severity."""
        counts = {s.value: 0 for s in Severity}
        for component in components:
            for issue in component.issues:
                counts[issue.severity.value] += 1
        return counts

    def _generate_recommendations(self, components: list[ComponentInfo]) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check test coverage
        untested = [c for c in components if not c.has_tests]
        if len(untested) > len(components) * 0.5:
            recommendations.append(
                f"Test coverage is low. {len(untested)} of {len(components)} "
                f"components lack tests."
            )

        # Check for components without selectors (for E2E testing)
        no_selectors = [c for c in components if not c.selectors]
        if no_selectors:
            recommendations.append(
                f"{len(no_selectors)} components lack test selectors (data-testid). "
                f"Consider adding them for reliable E2E testing."
            )

        return recommendations

    # Helper methods for subclasses

    def extract_imports(self, parsed: ParsedFile) -> list[str]:
        """Extract import statements from parsed file."""
        imports = []
        for imp in parsed.get_imports():
            imports.append(imp.text.strip())
        return imports

    def extract_exports(self, content: str) -> list[str]:
        """Extract exported names from file content."""
        exports = []

        # ES6 exports
        export_patterns = [
            r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)',
            r'export\s*\{\s*([^}]+)\s*\}',
            r'module\.exports\s*=\s*(\w+)',
        ]

        for pattern in export_patterns:
            for match in re.finditer(pattern, content):
                exports.extend(
                    name.strip() for name in match.group(1).split(",")
                )

        return exports

    def extract_selectors(self, content: str) -> list[str]:
        """Extract test selectors from content."""
        selectors = []

        # data-testid
        for match in re.finditer(r'data-testid=["\']([^"\']+)["\']', content):
            selectors.append(f"[data-testid='{match.group(1)}']")

        # data-test
        for match in re.finditer(r'data-test=["\']([^"\']+)["\']', content):
            selectors.append(f"[data-test='{match.group(1)}']")

        # aria-label
        for match in re.finditer(r'aria-label=["\']([^"\']+)["\']', content):
            selectors.append(f"[aria-label='{match.group(1)}']")

        return selectors

    def find_function_calls(self, node: ASTNode, function_name: str) -> list[ASTNode]:
        """Find all calls to a specific function in AST."""
        calls = []
        for call in node.find_descendants("call_expression"):
            if function_name in call.text:
                calls.append(call)
        return calls
